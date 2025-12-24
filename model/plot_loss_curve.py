import torch
import matplotlib.pyplot as plt

# 1. 設定你的模型檔案路徑
model_path = 'best_model_dropout_epoch220_testacc0.852.pth' 

try:
    # 2. 載入模型檔案
    data = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 3. 提取 train_acc_hist
    if isinstance(data, dict) and 'train_acc_hist' in data:
        train_acc_hist = data['train_acc_hist']
        
        # 4. 繪製 Loss Curve
        plt.figure(figsize=(12, 6))
        plt.plot(train_acc_hist, linewidth=2, color='#2E86AB')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 添加統計資訊
        min_loss = min(train_acc_hist)
        min_epoch = train_acc_hist.index(min_loss)
        plt.axhline(y=min_loss, color='red', linestyle='--', alpha=0.5, label=f'Min Loss: {min_loss:.4f} (Epoch {min_epoch})')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
        print(f"✓ Loss curve 已儲存為 'loss_curve.png'")
        print(f"✓ Total Epochs: {len(train_acc_hist)}")
        print(f"✓ Initial Loss: {train_acc_hist[0]:.4f}")
        print(f"✓ Final Loss: {train_acc_hist[-1]:.4f}")
        print(f"✓ Minimum Loss: {min_loss:.4f} at Epoch {min_epoch}")
        plt.show()
        
    else:
        print("Error: 'train_acc_hist' not found in the loaded model file.")
        
except FileNotFoundError:
    print(f"Error: File not found at '{model_path}'. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")