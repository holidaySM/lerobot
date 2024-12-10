#!/bin/bash

# Define the arrays of parameters
subset_fractions=(10 20 40 80 160 206)
vision_backbones=(resnet18 resnet50 resnet101)
policies=(diffusion)

# Other fixed parameters
env=pusht
training_batch_size=16
wandb_enable=true
wandb_project=lerobot
wandb_notes="pusht_holi"

# Loop over all combinations of parameters
for policy in "${policies[@]}"
do
    for vision_backbone in "${vision_backbones[@]}"
    do
        for subset_fraction in "${subset_fractions[@]}"
        do
            # Create a unique name for the experiment
            wandb_name="${policy}_${vision_backbone}_subset${subset_fraction}"

            subset_fraction_str="'train[:${subset_fraction}%]'"

            python lerobot/scripts/train.py \
                policy="$policy" \
                env="$env" \
                training.split="$subset_fraction_str" \
                training.batch_size="$training_batch_size" \
                policy.vision_backbone="$vision_backbone" \
                wandb.enable="$wandb_enable" \
                wandb.project="$wandb_project" \
                wandb.notes="$wandb_notes" \
                wandb.name="$wandb_name"
        done
    done
done


