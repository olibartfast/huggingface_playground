# huggingface_playground
Huggingface playground on models from https://huggingface.co/tasks

### Huggingface virtual env setup instructions

1. **Create and run the script**:
   - Make the huggingface setup script executable:
     ```bash
     chmod +x setup_huggingface_env.sh
     ```
   - Run the script:
     ```bash
     ./setup_huggingface_env.sh
     ```

2. **Activating the virtual environment**:
   After running the script, you can activate the virtual environment using:
   ```bash
   source ./huggingface_env/bin/activate
   ```

3. **Installing additional packages**:
   You can add more packages to the `pip install` command or manually edit the `requirements.txt` file.


### Hugging Face on Amazon SageMaker
https://huggingface.co/docs/sagemaker/index

### Hugging Face on Nvidia Triton Server
https://github.com/triton-inference-server/tutorials/tree/main/HuggingFace
