use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use candle_transformers::generation::LogitsProcessor;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::io::Write;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    
    println!("Loading Qwen2.5 0.5B Instruct (GGUF)...");
    let mut file = std::fs::File::open("qwen2.5-0.5b-instruct-q4_k_m.gguf")?;
    let model_content = gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(model_content, &mut file, &device)?;

    let tokenizer = Tokenizer::from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", None)?;
    let mut logits_processor = LogitsProcessor::new(42, Some(0.7), Some(0.9));
    
    // Initialize the TUI Editor
    let mut rl = DefaultEditor::new()?;
    println!("\n=== Rust LLM Engine TUI ===");
    println!("Type 'quit' or 'exit' to stop.\n");

    // We will keep the conversation history in this string
    let mut conversation_history = String::new();
    
    // The main TUI Chat Loop
    loop {
        let readline = rl.readline("You: ");
        match readline {
            Ok(line) => {
                let user_input = line.trim();
                if user_input == "quit" || user_input == "exit" {
                    break;
                }
                
                // Add input to Rustyline history so you can press 'Up' to see previous prompts
                rl.add_history_entry(user_input)?;
                
                // Append the new prompt in ChatML format
                conversation_history.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", user_input));
                
                let mut tokens = tokenizer.encode(conversation_history.clone(), false)?.get_ids().to_vec();
                
                print!("AI: ");
                std::io::stdout().flush()?;
                
                // Prefill Phase
                let input_tensor = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
                let logits = model.forward(&input_tensor, 0)?;
                let last_logits = logits.squeeze(0)?;
                
                let mut next_token = logits_processor.sample(&last_logits)?;
                tokens.push(next_token);
                
                // We keep track of the AI's raw response to append to the history later
                let mut ai_response = String::new();
                
                if let Ok(first_word) = tokenizer.decode(&[next_token], false) {
                    print!("{}", first_word);
                    ai_response.push_str(&first_word);
                    std::io::stdout().flush()?;
                }
                
                // Autoregressive Generation Loop
                for _ in 0..1024 {
                    let input_tensor = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                    let logits = model.forward(&input_tensor, tokens.len() - 1)?; 
                    let last_logits = logits.squeeze(0)?; 
                    
                    next_token = logits_processor.sample(&last_logits)?;
                    tokens.push(next_token);
                    
                    if next_token == 151645 { // <|im_end|>
                        break;
                    }
                    
                    if let Ok(next_word) = tokenizer.decode(&[next_token], false) {
                        print!("{}", next_word);
                        ai_response.push_str(&next_word);
                        std::io::stdout().flush()?;
                    }
                }
                
                // Append the AI's final response and closing tag to the ongoing history
                conversation_history.push_str(&ai_response);
                conversation_history.push_str("<|im_end|>\n");
                
                println!("\n");
            },
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                break;
            },
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
    
    println!("Engine shut down gracefully.");
    Ok(())
}
