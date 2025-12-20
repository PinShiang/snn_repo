
snn_repo 
 ┣ data 
   ┣ audios
   ┣ (various npz directory)
   ┣ convert.sh             #conversion script
   ┣ plot_longspike.py
   ┣ plot_spike1to10_100.py #一次plot0~9的npz，是digitize成100格後的
   ┗ plot_spike1to10.py     #一次plot0~9的npz
 ┣ model       
   ┣ (models)
   ┗ check_model.py       # 開pth檔檢查裡面的參數、training history
 ┣ inference.py           
 ┣ longAudioInfer.py     
 ┣ train.py               # sample code
 ┣ train_dropout.py       # Dropout (only eng)
 ┣ train_engdigit.py      # only train english
 ┣ train_engdigitshuffle.py
 ┣ utils.py               
 ┗ visualize.py           
