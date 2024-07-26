#!/bin/bash

mkdir pretrained
cd pretrained
wget https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth
wget https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth
wget https://huggingface.co/FoundationVision/var/resolve/main/var_d20.pth
wget https://huggingface.co/FoundationVision/var/resolve/main/var_d24.pth
wget https://huggingface.co/FoundationVision/var/resolve/main/var_d30.pth
wget https://huggingface.co/FoundationVision/var/resolve/main/var_d30.pth
cd ..

