use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, Embedding, LayerNorm, Linear, Module, VarBuilder,
};
use candle_transformers::generation::LogitsProcessor;
use std::io::Write;
use tokenizers::Tokenizer;

// --- CONFIGURATION ---
struct Config {
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    vocab_size: usize,
}

// --- MLP BLOCK ---
struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl MLP {
    fn load(vb: VarBuilder, config: &Config) -> candle_core::Result<Self> {
        let hidden_size = config.n_embd * 4;
        let c_fc = linear(config.n_embd, hidden_size, vb.pp("c_fc"))?;
        let c_proj = linear(hidden_size, config.n_embd, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.c_fc.forward(xs)?;
        let x = x.gelu()?;
        self.c_proj.forward(&x)
    }
}

// --- ATTENTION BLOCK ---
struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
}

impl CausalSelfAttention {
    fn load(vb: VarBuilder, config: &Config) -> candle_core::Result<Self> {
        let c_attn = linear(config.n_embd, 3 * config.n_embd, vb.pp("c_attn"))?;
        let c_proj = linear(config.n_embd, config.n_embd, vb.pp("c_proj"))?;
        Ok(Self { c_attn, c_proj, n_head: config.n_head })
    }

    // CUSTOM FORWARD PASS: Now accepts a mutable cache
    fn forward(&self, xs: &Tensor, cache: &mut Option<(Tensor, Tensor)>) -> candle_core::Result<Tensor> {
        let (b_sz, seq_len, n_embd) = xs.dims3()?;
        let head_dim = n_embd / self.n_head;

        let qkv = self.c_attn.forward(xs)?;
        let q = qkv.narrow(2, 0, n_embd)?;
        let k = qkv.narrow(2, n_embd, n_embd)?;
        let v = qkv.narrow(2, 2 * n_embd, n_embd)?;

        let q = q.reshape((b_sz, seq_len, self.n_head, head_dim))?.transpose(1, 2)?.contiguous()?;
        let mut k = k.reshape((b_sz, seq_len, self.n_head, head_dim))?.transpose(1, 2)?.contiguous()?;
        let mut v = v.reshape((b_sz, seq_len, self.n_head, head_dim))?.transpose(1, 2)?.contiguous()?;

        // KV CACHE LOGIC
        if let Some((past_k, past_v)) = cache {
            k = Tensor::cat(&[past_k, &k], 2)?.contiguous()?;
            v = Tensor::cat(&[past_v, &v], 2)?.contiguous()?;
        }
        
        // Save current state to cache
        *cache = Some((k.clone(), v.clone()));
        
        let seq_total = k.dim(2)?;

        let att = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let att = (att / (head_dim as f64).sqrt())?;

        // Dynamic Causal Masking
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_total).map(move |j| if j > i + (seq_total - seq_len) { -1e9_f32 } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (seq_len, seq_total), xs.device())?;
        
        let att = att.broadcast_add(&mask)?;
        let att = candle_nn::ops::softmax(&att, 3)?;

        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b_sz, seq_len, n_embd))?;

        self.c_proj.forward(&y)
    }
}

// --- TRANSFORMER BLOCK ---
struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl Block {
    fn load(vb: VarBuilder, config: &Config) -> candle_core::Result<Self> {
        let ln_1 = layer_norm(config.n_embd, 1e-5, vb.pp("ln_1"))?;
        let attn = CausalSelfAttention::load(vb.pp("attn"), config)?;
        let ln_2 = layer_norm(config.n_embd, 1e-5, vb.pp("ln_2"))?;
        let mlp = MLP::load(vb.pp("mlp"), config)?;
        Ok(Self { ln_1, attn, ln_2, mlp })
    }

    fn forward(&self, xs: &Tensor, cache: &mut Option<(Tensor, Tensor)>) -> candle_core::Result<Tensor> {
        let x = self.ln_1.forward(xs)?;
        let x = self.attn.forward(&x, cache)?;
        let x = (xs + x)?;
        
        let m = self.ln_2.forward(&x)?;
        let m = self.mlp.forward(&m)?;
        
        Ok((x + m)?)
    }
}

// --- MAIN GPT SHELL ---
struct GPT {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl GPT {
    fn load(vb: VarBuilder, config: &Config) -> candle_core::Result<Self> {
        let transformer_vb = vb.pp("transformer");
        let wte = embedding(config.vocab_size, config.n_embd, transformer_vb.pp("wte"))?;
        let wpe = embedding(1024, config.n_embd, transformer_vb.pp("wpe"))?;
        
        let mut blocks = Vec::new();
        let h_vb = transformer_vb.pp("h");
        for i in 0..config.n_layer {
            blocks.push(Block::load(h_vb.pp(&i.to_string()), config)?);
        }
        
        let ln_f = layer_norm(config.n_embd, 1e-5, transformer_vb.pp("ln_f"))?;
        let lm_head = linear_no_bias(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;
        
        Ok(Self { wte, wpe, blocks, ln_f, lm_head })
    }

    // Now accepts a vector of caches (one for each block)
    fn forward(&self, xs: &Tensor, caches: &mut Vec<Option<(Tensor, Tensor)>>) -> candle_core::Result<Tensor> {
        let (_b_sz, seq_len) = xs.dims2()?;
        
        // Determine offset for positional embeddings based on cache size
        let past_len = if let Some((k, _)) = &caches[0] { k.dim(2)? } else { 0 };
        
        let pos = Tensor::arange(past_len as u32, (past_len + seq_len) as u32, xs.device())?;
        
        let tok_emb = self.wte.forward(xs)?;
        let pos_emb = self.wpe.forward(&pos)?.unsqueeze(0)?;
        
        let mut x = tok_emb.broadcast_add(&pos_emb)?;
        
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, &mut caches[i])?;
        }
        
        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }
}

// --- EXECUTION ENGINE ---
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let device = Device::Cpu;
    let file_path = "nanoGPT.safetensors";
    
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[file_path], candle_core::DType::F32, &device)? };
    let config = Config { n_embd: 768, n_head: 12, n_layer: 12, vocab_size: 50257 };
    let model = GPT::load(vb, &config)?;

    let tokenizer = Tokenizer::from_pretrained("gpt2", None)?;
    // Generate a random seed based on the current system time
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let mut logits_processor = LogitsProcessor::new(seed, Some(0.8), None);
    
    let prompt = "Why did the AI cross the road?";
    let mut tokens = tokenizer.encode(prompt, false)?.get_ids().to_vec();

    print!("\nPrompt: {}\nOutput: ", prompt);
    std::io::stdout().flush()?;

    // Initialize 12 empty caches for our 12 transformer blocks
    let mut caches = vec![None; config.n_layer];

    for _ in 0..50 {
        // If cache exists, ONLY pass the very last token
        let input_tokens = if caches[0].is_some() {
            vec![*tokens.last().unwrap()]
        } else {
            tokens.clone()
        };
        
        let input_tensor = Tensor::new(input_tokens.as_slice(), &device)?.unsqueeze(0)?;
        
        // Forward pass now mutates the cache in-place
        let logits = model.forward(&input_tensor, &mut caches)?;
        
        let (_b, seq, _v) = logits.dims3()?;
        let last_logits = logits.i((0, seq - 1, ..))?;
        
        let next_token = logits_processor.sample(&last_logits)?;
        tokens.push(next_token);
        
        let next_word = tokenizer.decode(&[next_token], false)?;
        print!("{}", next_word);
        std::io::stdout().flush()?;
    }
    
    println!("\n\n✅ Generation Complete.");
    Ok(())
}
