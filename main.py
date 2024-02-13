
import json
import os
import pickle

class Main(object):
   def __init__(self):
      from llama_cpp import Llama
      model_filepath = "mistral-7b-instruct-v0.1.Q5_0.gguf"

      # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
      self.llm = Llama(
      model_path=model_filepath,  # Download the model file first
      n_ctx=16384,  # The max sequence length to use - note that longer sequence lengths require much more resources
      n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
      n_gpu_layers=0         # The number of layers to offload to GPU, if you have GPU acceleration available
      ) 
      
   def predict(self, skill_input):
      try:
         output = self.llm(
         skill_input, # AQUÍ SE ESCRIBE EL PROMPT
         max_tokens=512,  # Generate up to 512 tokens
         stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
         echo=True        # Whether to echo the prompt
         )
         return json.dumps(output["choices"][0]["text"])
      except Exception as e:
         return "Error during prediction"
      
       
if __name__ == '__main__':
   a = Main()
   results = a.predict(skill_input="How hot is the surface of the Sun")
   print(results)