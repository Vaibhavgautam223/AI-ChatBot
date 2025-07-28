from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def initialize_models():
    """Initialize both chatbot models with error handling"""
    try:
        # Initialize ChatterBot
        bot = ChatBot('MyBot')
        trainer = ChatterBotCorpusTrainer(bot)
        trainer.train('chatterbot.corpus.english')
        
        # Initialize DialoGPT
        print("Loading DialoGPT model (this may take a moment)...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        
        return bot, model, tokenizer
        
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        raise

def chat_with_chatterbot(bot):
    """Chat loop for ChatterBot"""
    print("\nChatterBot activated. Type 'exit' to end or 'switch' to change models.")
    while True:
        try:
            text = input("You: ")
            if text.lower() == 'exit':
                return True  # Signal to exit completely
            if text.lower() == 'switch':
                return False  # Signal to switch models
                
            response = bot.get_response(text)
            print(f"Bot: {response}")
            
        except Exception as e:
            print(f"Error in conversation: {str(e)}")
            continue

def chat_with_dialoGPT(model, tokenizer, chat_history_ids=None):
    """Chat loop for DialoGPT"""
    print("\nDialoGPT activated. Type 'exit' to end or 'switch' to change models.")
    while True:
        try:
            text = input("You: ")
            if text.lower() == 'exit':
                return True  # Signal to exit completely
            if text.lower() == 'switch':
                return False  # Signal to switch models
                
            # Encode the new user input, add the eos_token and return a tensor in Pytorch
            new_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
            
            # Append the new input to the chat history (if it exists)
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
            
            # Generate response
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8
            )
            
            # Decode the response
            response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            print(f"Bot: {response}")
            
        except Exception as e:
            print(f"Error in conversation: {str(e)}")
            continue

def main():
    """Main function to run the chatbot application"""
    try:
        bot, model, tokenizer = initialize_models()
        current_model = "chatterbot"  # Start with ChatterBot
        chat_history_ids = None
        
        print("Chatbot initialized. You can switch between models by typing 'switch'.")
        
        while True:
            if current_model == "chatterbot":
                should_exit = chat_with_chatterbot(bot)
                if should_exit:
                    break
                current_model = "dialoGPT"
            else:
                should_exit = chat_with_dialoGPT(model, tokenizer, chat_history_ids)
                if should_exit:
                    break
                current_model = "chatterbot"
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        print("\nChat session ended.")

if __name__ == "__main__":
    main()
# This code combines a simple quiz game with a chatbot application using ChatterBot and DialoGPT.