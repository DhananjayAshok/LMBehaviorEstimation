# Language Model Behavior Anticipation
We show that probing the internal states of the input tokens to a LM often yeilds success at predicting the downstream behavior of the output. When wrapped up in a conformal prediction framework, this signal can be used to create a trustworthy and useful early warning system. 

This code base will allow you to replicate our results, easily extend our method, or apply it to new datasets/tasks. 

## Repository Layout

There are three key steps in this project:
1. Download datasets and set them up for task specific generation ([data.py](data.py))
2. Generate the hidden states on all splits of the data ([save_inference.py](save_inference.py))
3. Fit the Conformal Probes onto the hidden states ([modeling.py](modeling.py))

## Installation and Setup
First, install the required packages with:
```bash
pip install -r requirements.txt
```

Next, set the required arguments in [proj_params.sh](proj_params.sh)

You can now start setting up datasets with
```bash
bash scripts/get_data.sh
```
Finally, generate the task data with 
```bash
python data.py
```
## Running Experiments

Once the data is set up, you can generate hidden states with (check and adapt before you run):
```bash
bash scripts/inference.sh
```

Then, you can fit the conformal probes and see the results with:
```bash
bash scripts/modeling.sh
```

You can edit the modeling.sh file to change the layer used to probe, or the target property (see label.py)