import argparse
from typing import List, Optional
import requests
import json
import time
from tqdm import tqdm
from transformers import AutoTokenizer

class TogetherAIInferenceEngine:
    def __init__(self, args):
        """Initialize Together AI inference engine"""
        self.args = args
        
        # Verify API key is provided
        if not hasattr(args, 'together_api_key') or not args.together_api_key:
            raise RuntimeError("Together API key is required. Set args.together_api_key.")
            
        self.api_key = args.together_api_key
        self.api_url = "https://api.together.xyz/v1/completions"
        
        # Setup tokenizer for chat templating if needed
        if hasattr(args, 'use_chat_template') and args.use_chat_template:
            tokenizer_name = args.tokenizer if hasattr(args, 'tokenizer') else args.model
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        else:
            self.tokenizer = None
            
        # Configure generation parameters
        self.generation_params = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stop": getattr(args, 'stop_tokens', []),
        }
        
        print(f"Initialized Together AI engine for model: {args.model}")

    def format_chat_prompt(self, messages: List[dict]) -> Optional[str]:
        """Format messages using chat template"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available for applying chat template")
        
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            raise ValueError("Tokenizer does not support chat templates")
            
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if hasattr(self.args, 'max_prompt_length') and self.args.max_prompt_length is not None:
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            if len(tokenized.input_ids[0]) > self.args.max_prompt_length:
                return " "
        
        return prompt
            
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate completions via Together AI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        results = []
        
        # Process prompts individually since Together AI completions endpoint 
        # expects a single prompt string, not a batch
        for i, prompt in enumerate(tqdm(prompts, desc="Generating responses", disable=not self.args.debug)):
            
            payload = {
                "model": self.args.model,
                "prompt": prompt,  # Single string, not a list
                **self.generation_params
            }
            
            success = False
            for attempt in range(3):
                try:
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Handle single response
                    if isinstance(result, dict) and 'choices' in result and len(result['choices']) > 0:
                        generated_text = result['choices'][0]['text'].strip()
                        results.append(generated_text)
                        success = True
                        if self.args.debug:
                            print(f"Successfully generated response for prompt {i+1}")
                        break
                    else:
                        print(f"Unexpected response format for prompt {i+1}: {result}")
                        results.append("")
                        success = True
                        break
                        
                except requests.exceptions.HTTPError as e:
                    print(f"Attempt {attempt + 1} failed for prompt {i+1}: {prompt[:50]}...")
                    print(f"HTTP Error: {str(e)}")
                    
                    # Try to get more detailed error information
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            error_detail = e.response.json()
                            print(f"Error details: {json.dumps(error_detail, indent=2)}")
                        except:
                            print(f"Response text: {e.response.text}")
                    
                    if attempt == 2:
                        print(f"Final failure for prompt {i+1}")
                        results.append("")  # Add empty result for failed prompt
                    else:
                        time.sleep(2**attempt)  # Exponential backoff
                        
                except requests.exceptions.RequestException as e:
                    print(f"Request error on attempt {attempt + 1} for prompt {i+1}: {str(e)}")
                    if attempt == 2:
                        results.append("")  # Add empty result for failed prompt
                    else:
                        time.sleep(2**attempt)
                        
                except Exception as e:
                    print(f"Unexpected error on attempt {attempt + 1} for prompt {i+1}: {str(e)}")
                    if attempt == 2:
                        results.append("")  # Add empty result for failed prompt
                    else:
                        time.sleep(2**attempt)
            
            # Add a small delay between requests to be respectful to the API
            if i < len(prompts) - 1:  # Don't delay after the last request
                time.sleep(0.1)
        
        return results

def main():
    # Set up arguments
    parser = argparse.ArgumentParser(description="Together AI Inference Engine")
    parser.add_argument("--together_api_key", type=str, default="a6b3082f044f6fc0ebd8932e926428b897635b31606012b190f6c19fab62df32", help="Your Together API key")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Model name")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference (not used with current implementation)")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling value")
    parser.add_argument("--stop_tokens", nargs="*", default=["</s>", "\n\n"], help="Stop tokens")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--use_chat_template", action="store_true", help="Use chat template for formatting")
    parser.add_argument("--tokenizer", type=str, default="", help="Tokenizer name (if different from model)")
    parser.add_argument("--max_prompt_length", type=int, default=None, help="Maximum prompt length in tokens")
    
    args = parser.parse_args()
    
    # Check if API key is set
    if not args.together_api_key:
        print("Error: Please provide your Together API key using --together_api_key")
        print("You can get your API key from: https://api.together.xyz/settings/api-keys")
        return
    
    try:
        # Initialize inference engine
        engine = TogetherAIInferenceEngine(args)
        
        # Create test prompts
        prompts = [
            "Translate 'Hello, world!' to Chinese:",
            "Translate 'Hello, world!' to French:"
        ]
        
        # Generate responses
        print(f"Generating responses for {len(prompts)} prompts...")
        responses = engine.generate(prompts)
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            print(f"\n[Prompt {i+1}]: {prompt}")
            print(f"[Response {i+1}]: {response if response else '(No response generated)'}")
            print("-" * 50)
            
        # Summary
        successful_responses = sum(1 for r in responses if r.strip())
        print(f"\nSummary: {successful_responses}/{len(prompts)} responses generated successfully")
        
    except Exception as e:
        print(f"Error initializing engine: {str(e)}")
        return

if __name__ == "__main__":
    main()