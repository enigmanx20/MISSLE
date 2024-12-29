# **MISSLE**

**M**elanoma **I**ndicative **S**corer for **S**entinel **L**ymph node **E**valuation (MISSLE) is an ensemble gradient boosting model combining multiple instance deep learning ([CLAM](https://github.com/mahmoodlab/CLAM/tree/master)) and the clinical nomogram ([https://www.mskcc.org/nomograms/melanoma/sentinel_lymph_node_metastasis](https://www.mskcc.org/nomograms/melanoma/sentinel_lymph_node_metastasis)). 

### Requirements

Install required packages according [CLAM](https://github.com/mahmoodlab/CLAM/tree/master) instruction for whole slide image segmentation, patchfying, and feature extraction. MISSLE requires the additional packages as follows:

```
pytorch>=1.8.0 (idealy, pytorch==2.3.0)
scikit-learn==1.0.1
numpy
pandas
matplotlib (for paper reproduction)
shap==0.44.1 (for paper reproduction)
```

### How to use the pre-trained MISSLE model

1. Create patches and ResNet50trunc embeddings (feature extraction) on the melanoma WSIs according to the [CLAM](https://github.com/mahmoodlab/CLAM/tree/master) instruction. Calculate the clinical nomogram probablities at [https://www.mskcc.org/nomograms/melanoma/sentinel_lymph_node_metastasis](https://www.mskcc.org/nomograms/melanoma/sentinel_lymph_node_metastasis).
2. Load CLAM-R50 model and gradient boosting model. Calculate MISSLE probability as `missle_score`.

```
import torch
from models.model_clam import CLAM_SB
import pickle

# initialize CLAM-SB and load chekpoint file CLAM-R50.pt.
model = CLAM_SB(gate=True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
                 instance_loss_fn=torch.nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024)
model.load_state_dict(torch.load('CLAM-R50.pt'), map_location='cpu')
model.eval()

embedding = torch.zeros(1, BAG_SIZE, 1024) # load your ResNet50trunc embedding data

_, clam_prob, _, _, _ = CLAM_SB(embedding, label=None, instance_eval=False, return_features=False, attention_only=False)

### prepare your data here###
### below is the dummy code##
# the clam_probs and nomogram_probs must have the same dimensions.
clam_probs = [0.2, 0.6, 0.3, ...]
nomogram_probs = [0.5, 0.1, 0.2, ...]
#############################

# load missle model from missle.pickle.
with open('missle.pickle', 'rb') as f:
    missle = pickle.load(f)

missle_probs = missle.predict_proba(np.stack([clam_prob, nomogram_prob], axis=1))[:, 1]
```

### How to make MISSLE model on your data

1. Create patches and embeddings (ResNet50trunc or UNI) on the melanoma WSIs according to the [CLAM](https://github.com/mahmoodlab/CLAM/tree/master) instruction. Calculate the clinical nomogram probablities at [https://www.mskcc.org/nomograms/melanoma/sentinel_lymph_node_metastasis](https://www.mskcc.org/nomograms/melanoma/sentinel_lymph_node_metastasis).

   Our snippet for patchfying is here.
   `python create_patches_fp.py --source /YOUR_WSI_FOLDER --save_dir /YOUR_PATCH_FOLDER --patch_size 256 --seg --patch --stich --preset ./presets/tcga.csv`

   When applying XML annotation files created with [ASAP](https://github.com/computationalpathologygroup/ASAP), use the following command instead of `create_patches_fp.py`. Annotation files should be located as the following.

   ```
   --WSI_FOLDER--001.svs
               --002.svs
               --annotations--001.xml
                            --002.xml
   ```

    Run the following snippet.

`python create_patches_fp_anno.py --source /YOUR_WSI_FOLDER --save_dir /YOUR_PATCH_FOLDER --patch_size 256 --seg`

2. Prepare the csv file (with the label of positive/negative and train/val/test) folliwing CLAM instructions and change the `csv_path` in `main.py` line 163.

   ```
   dataset = Generic_MIL_Dataset(csv_path = '/YourDataPath/positive_vs_negative.csv',
                               data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                               shuffle = False, 
                               seed = args.seed, 
                               print_info = True,
                               label_dict = {'negative':0, 'positive':1},
                               patient_strat=False,
                               ignore=[])
   ```
3. Train CLAM-SB with the following snippet. Make sure to change the `./DATA_ROOT_DIR` of your own.

   `python main.py --drop_out 0.25 --lr 2e-3 --k 1 --exp_code task_1_tumor_vs_normal_CLAM_50 --weighted_sample --bag_loss ce --inst_loss ce --task task_1_tumor_vs_normal --model_type clam_sb --model_size small --log_data --data_root_dir ./DATA_ROOT_DIR --embed_dim 1024 --max_epochs 30 --warmup_cosine -—opt adam --bag_weight 0.7 --B 8`

   When training annotated region patch embeddings, use the following snippet.

   `python main.py --drop_out 0.25 --lr 2e-3 --k 1 --exp_code task_1_tumor_vs_normal_CLAM_50 --weighted_sample --bag_loss ce --inst_loss ce --task task_1_tumor_vs_normal --model_type clam_sb --model_size small --log_data --data_root_dir ./DATA_ROOT_DIR --embed_dim 1024 --max_epochs 50 --warmup_cosine -—opt adam --bag_weight 0.7 --B 4`

   Your CLAM model is saved under the `./DATA_ROOT_DIR/results/task_1_tumor_vs_normal_CLAM_50_s1` as `s_0_checkpoint.pt.`
4. Make gradient boosting model from CLAM and nomogram probabilities.

```
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
# prepare training and test data
X_train = train_data[['CLAM_prob', 'Nomogram_prob']]
y_train = train_data.loc[X_train.index, 'LN_metastasis_binary']
X_test = test_data[['CLAM_prob', 'Nomogram_prob']]
y_test = test_data.loc[X_test.index, 'LN_metastasis_binary']

missle = GradientBoostingClassifier(random_state=42)
# train the model on training data
missle.fit(X_train, y_train)

#predict on test data
missle_prob = missle.predict_proba(X_test)[:, 1]
```

## **Liscence**

This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

## Reference

Ahead of printing.
