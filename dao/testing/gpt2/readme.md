Preequisite
```
ln -s  ~/ssd/workspace/DAO/models models
pip install -r requirements.txt 
```

Script 
```
# training, no flash attention  
python ./gpt2_torch_dao.py --model_size 124M --batch_size 2
# training, break after 10 iteration 
python ./gpt2_torch_dao.py --model_size 124M --batch_size 2
```