import subprocess

# Use subprocess.run for better practice
subprocess.run(['curl', '-L', 'https://huggingface.co/datasets/khushwant04/Research-Papers/resolve/main/research-papers.tar?download=true', '-o', 'research-papers.tar'], check=True)
subprocess.run(['tar', '-xf', 'research-papers.tar'], check=True)
