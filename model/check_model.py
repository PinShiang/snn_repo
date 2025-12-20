import torch

# 1. Set your file path here
# Example: 'model/best_model_dropout_epoch10_testacc0.950.pth'
model_path = 'best_model_dropout_epoch350_testacc0.840.pth' 

try:
    # 2. Load the file
    # map_location='cpu' ensures it loads even without a GPU
    data = torch.load(model_path, map_location='cpu', weights_only=False)

    print(f"Successfully loaded! Data type: {type(data)}\n")

    # 3. Iterate and print content
    if isinstance(data, dict):
        # Print Header
        print(f"{'Key':<25} | {'Value / Shape'}")
        print("-" * 70)
        
        for key, value in data.items():
            # Case A: PyTorch Tensor (e.g., Weights w1, w2)
            if torch.is_tensor(value):
                print(f"{key:<25} | Tensor Shape: {value.shape}")
            
            # Case B: Long Lists (e.g., loss history)
            elif isinstance(value, list) and len(value) > 10:
                print(f"{key:<25} | List (Length: {len(value)}), First 5: {value[:5]}...")
            
            # Case C: Scalar values or small parameters
            else:
                print(f"{key:<25} | {value}")
                
    else:
        # Fallback if the file content is not a dictionary
        print("The loaded data is not a dictionary. Raw content:")
        print(data)

except FileNotFoundError:
    print(f"Error: File not found at '{model_path}'. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")