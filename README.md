> [**ControlVAR: Exploring Controllable Visual Autoregressive Modeling**](https://arxiv.org/pdf/2406.09750)
>
> Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Zhe Lin, Rita Singh, Bhiksha Raj

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.02905-b31b1b.svg)](https://arxiv.org/pdf/2406.09750)&nbsp;

<p align="center"><img src="Illustration.png" width="700"/></p>

# Updates
- **(2024-10-10)** Try our new image tokenizer [ImageFolder](https://lxa9867.github.io/works/imagefolder/index.html)🚀 for faster speed🔥, shorter length🔥 and better FID🔥!
- **(2024-08-23)** We released pretrained checkpoints.
- **(2024-07-28)** We begin to upload the dataset (~400G) to [hugging-face](https://huggingface.co/datasets/qiuk6/ImageNet2012_condition) 🤗. 
- **(2024-07-26)** We released the code for Intel HPU training (GPU version is compatible). 
- **(2024-07-25)** Repo created. The code and datasets will be released in two weeks.


# Setup

Get pre-trained VQVAE from VAR.
```
mkdir pretrained
cd pretrained
wget https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth
```
Install required packages.
```
pip install -r requirements.txt
```


# Dataset

The pseudo-labeled ImageNet dataset (mask, canny, depth, and normal) is available at [hugging-face](https://huggingface.co/datasets/qiuk6/ImageNet2012_condition) 🤗. Please download the original ImageNet2012 dataset from [official website](https://www.image-net.org/) and arrange the files in the following format.


```
ImageNet2012
├── train
├── val
├── train_canny
├── train_mask
├── train_normal
├── train_depth
├── val_canny
├── val_mask
├── val_normal
└── val_depth
```

We provide the example function to convert parquet to data for our dataset in huggingface:
```
# Function to convert Parquet to images or JSON files
def convert_parquet_to_images_or_json(parquet_file_path, output_dir):
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file_path)

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the filename and check if it's image or JSON data
        filename = row['filename']

        # Check if 'image_data' column exists in the DataFrame
        if 'image_data' in df.columns:
            image_data = row['image_data']

            # Convert the binary data back to an image
            image = Image.open(io.BytesIO(image_data))

            # Determine the full output path
            full_output_path = os.path.join(output_dir, filename)

            # Create directories if they do not exist
            os.makedirs(os.path.dirname(full_output_path), exist_ok=True)

            # Save the image using the original filename
            image.save(full_output_path)
            print(f"Saved image: {full_output_path}")

        # Check if 'json_data' column exists in the DataFrame
        elif 'json_data' in df.columns:
            json_data = row['json_data']
            json_data = convert_numpy_to_list(json_data)
            # Determine the full output path
            full_output_path = os.path.join(output_dir, filename)

            # Create directories if they do not exist
            os.makedirs(os.path.dirname(full_output_path), exist_ok=True)

            # Save the JSON data to a file
            with open(full_output_path, 'w') as json_file:
                json.dump(json_data, json_file)
            print(f"Saved JSON: {full_output_path}")

        else:
            print(f"Unknown data format in {parquet_file_path}. Skipping row {index}.")

def convert_numpy_to_list(obj):
    """
    Recursively convert numpy arrays in a nested dictionary or list to lists.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return convert_numpy_to_list(obj.tolist())
    else:
        return obj

# Example usage
parquet_dir = "./downloaded_parquet_files"
output_image_dir = "."
os.makedirs(output_image_dir, exist_ok=True)

# List all Parquet files in the directory
parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]

# Convert all Parquet files back to images or JSON
for parquet_file in parquet_files:
    parquet_file_path = os.path.join(parquet_dir, parquet_file)
    print(f"Processing {parquet_file_path}...")
    convert_parquet_to_images_or_json(parquet_file_path, output_image_dir)
```

# Pretrained models
ID | Depth | Joint
--- |:---: |:---:
1 | 12 | [d12.pth](https://huggingface.co/qiuk6/ControlVAR/resolve/main/d12.pth)
2 | 16 | [d16.pth](https://huggingface.co/qiuk6/ControlVAR/resolve/main/d16.pth)
3 | 20 | [d20.pth](https://huggingface.co/qiuk6/ControlVAR/resolve/main/d20.pth)
4 | 24 | [d24.pth](https://huggingface.co/qiuk6/ControlVAR/resolve/main/d24.pth)
5 | 30 | [d30.pth](https://huggingface.co/qiuk6/ControlVAR/resolve/main/d30.pth)

# Train

```sh
python3 train_control_var_hpu.py
--batch_size $bs
--dataset_name imagenetC
--data_dir $path_to_ImageNetC
--gpus $gpus
--output_dir $output_dir
--multi_cond True
--config configs/train_mask_var_ImageNetC_d12.yaml
--var_pretrained_path pretrained/var_d12.pth
```

# Inference
```angular2html
python3 train_control_var_hpu.py
--batch_size $bs
--dataset_name imagenetC
--data_dir $path_to_ImageNetC
--gpus $gpus
--output_dir $output_dir
--multi_cond True
--val_only True
--resume $ckpt_path
```

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@article{li2024controlvar,
  title={ControlVAR: Exploring Controllable Visual Autoregressive Modeling},
  author={Li, Xiang and Qiu, Kai and Chen, Hao and Kuen, Jason and Lin, Zhe and Singh, Rita and Raj, Bhiksha},
  journal={arXiv preprint arXiv:2406.09750},
  year={2024}
}
```
