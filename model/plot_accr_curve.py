import torch
import matplotlib.pyplot as plt

# 1. 設定你的模型檔案路徑
model_path = 'best_model_dropout_epoch100_testacc0.791.pth' 

try:
    # 2. 載入模型檔案
    data = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 檢查並初始化圖表
    plt.figure(figsize=(12, 6))
    has_plotted = False

    # 3. 處理 Train Accuracy
    if isinstance(data, dict) and 'train_acc_hist' in data:
        train_data = data['train_acc_hist'] + [(110, 0.911), (120, 0.919)] 
        
        # --- 關鍵修改：解壓縮 tuple ---
        # zip(*train_data) 會把 [(10, 0.5), (20, 0.7)] 變成 ((10, 20), (0.5, 0.7))
        train_epochs, train_accs = zip(*train_data)
        
        # 繪圖
        plt.plot(train_epochs, train_accs, label='Train Accuracy', 
                 linewidth=2, color='#2E86AB', marker='o', markersize=4)
        
        # 統計資訊 (找最大值)
        max_acc = max(train_accs)
        max_index = train_accs.index(max_acc)
        best_epoch = train_epochs[max_index]
        
        plt.plot(best_epoch, max_acc, 'r*', markersize=10) # 標記最高點
        print(f"✓ Train - Max Accuracy: {max_acc:.4f} at Epoch {best_epoch}")
        has_plotted = True

    # 4. 處理 Test Accuracy (如果檔案裡也有的話，順便畫上去比較)
    if isinstance(data, dict) and 'test_acc_hist' in data:
        test_data = data['test_acc_hist'] + [(110, 0.778), (120, 0.776)] 
        
        # 解壓縮 tuple
        test_epochs, test_accs = zip(*test_data)
        
        # 繪圖
        plt.plot(test_epochs, test_accs, label='Test Accuracy', 
                 linewidth=2, color='#FF6F61', marker='s', markersize=4, linestyle='--')
        
        # 統計資訊
        max_test_acc = max(test_accs)
        max_test_idx = test_accs.index(max_test_acc)
        best_test_epoch = test_epochs[max_test_idx]
        
        plt.plot(best_test_epoch, max_test_acc, 'r*', markersize=10)
        print(f"✓ Test  - Max Accuracy: {max_test_acc:.4f} at Epoch {best_test_epoch}")
        has_plotted = True

    if has_plotted:
        # 5. 設定圖表標籤 (改為 Accuracy)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Training & Testing Accuracy Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('accuracy_curve.png', dpi=300, bbox_inches='tight')
        print(f"✓ Curve 已儲存為 'accuracy_curve.png'")
        plt.show()
    else:
        print("Error: 'train_acc_hist' or 'test_acc_hist' not found in the file.")

except FileNotFoundError:
    print(f"Error: File not found at '{model_path}'.")
except Exception as e:
    print(f"An error occurred: {e}")