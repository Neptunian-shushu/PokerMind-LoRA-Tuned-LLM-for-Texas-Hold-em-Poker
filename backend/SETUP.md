# SETUP fastapi 

```bash
# change to gpu node
srun --partition=ice-gpu --gres=gpu:H100:1 --mem=40G --time=01:00:00 --pty bash 

module load anaconda3
conda activate cs6220 

# run uvicorn in the background
nohup uvicorn aiService:app --host 0.0.0.0 --port 8000 > fastapi.log 2>&1 &

# test
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "instruction": "Your prompt here."
         }'

# kill the process
pkill -f uvicorn

```


Connect Frontend with Backend
```bash
# In your local computer
# This command creates an SSH tunnel that maps your local port 8001 to
# port 8000 on the remote GPU node (e.g. atl1-1-03-013-3-0) through the ICE login node.
ssh -L 8001:atl1-1-03-012-18-0:8000 mjin93@login-ice.pace.gatech.edu
npm run dev
```