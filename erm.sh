#!/usr/bin/env bash
# ResNet50, PACS
CUDA_VISIBLE_DEVICES=0 python erm.py data/PACS -d PACS -s A C S -t P -a resnet50 --freeze-bn --seed 0 --log logs/erm/PACS_P &
CUDA_VISIBLE_DEVICES=0 python erm.py data/PACS -d PACS -s P C S -t A -a resnet50 --freeze-bn --seed 0 --log logs/erm/PACS_A &
CUDA_VISIBLE_DEVICES=1 python erm.py data/PACS -d PACS -s P A S -t C -a resnet50 --freeze-bn --seed 0 --log logs/erm/PACS_C &
CUDA_VISIBLE_DEVICES=1 python erm.py data/PACS -d PACS -s P A C -t S -a resnet50 --freeze-bn --seed 0 --log logs/erm/PACS_S

# ResNet50, Office-Home
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a resnet50 --seed 0 --log logs/erm/OfficeHome_Pr &
CUDA_VISIBLE_DEVICES=0 python erm.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a resnet50 --seed 0 --log logs/erm/OfficeHome_Rw &
CUDA_VISIBLE_DEVICES=1 python erm.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl -a resnet50 --seed 0 --log logs/erm/OfficeHome_Cl &
CUDA_VISIBLE_DEVICES=1 python erm.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar -a resnet50 --seed 0 --log logs/erm/OfficeHome_Ar

# ResNet50, DomainNet
CUDA_VISIBLE_DEVICES=0 python erm.py data/domainnet -d DomainNet -s i p q r s -t c -a resnet50 --seed 0 --log logs/erm/DomainNet_c &
CUDA_VISIBLE_DEVICES=0 python erm.py data/domainnet -d DomainNet -s c p q r s -t i -a resnet50 --seed 0 --log logs/erm/DomainNet_i &
CUDA_VISIBLE_DEVICES=0 python erm.py data/domainnet -d DomainNet -s c i q r s -t p -a resnet50 --seed 0 --log logs/erm/DomainNet_p &
CUDA_VISIBLE_DEVICES=1 python erm.py data/domainnet -d DomainNet -s c i p r s -t q -a resnet50 --seed 0 --log logs/erm/DomainNet_q &
CUDA_VISIBLE_DEVICES=1 python erm.py data/domainnet -d DomainNet -s c i p q s -t r -a resnet50 --seed 0 --log logs/erm/DomainNet_r &
CUDA_VISIBLE_DEVICES=1 python erm.py data/domainnet -d DomainNet -s c i p q r -t s -a resnet50 --seed 0 --log logs/erm/DomainNet_s

# ResNet50, VLCS
CUDA_VISIBLE_DEVICES=0 python erm.py data/VLCS -d VLCS -s L S V -t C -a resnet50 --seed 0 --log logs/erm/VLCS_C &
CUDA_VISIBLE_DEVICES=0 python erm.py data/VLCS -d VLCS -s L S V -t L -a resnet50 --seed 0 --log logs/erm/VLCS_L &
CUDA_VISIBLE_DEVICES=1 python erm.py data/VLCS -d VLCS -s L S V -t S -a resnet50 --seed 0 --log logs/erm/VLCS_S &
CUDA_VISIBLE_DEVICES=1 python erm.py data/VLCS -d VLCS -s L S V -t V -a resnet50 --seed 0 --log logs/erm/VLCS_V

# ResNet50, Terra
CUDA_VISIBLE_DEVICES=0 python erm.py data/Terra -d Terra -s 38 43 46 -t 100 -a resnet50 --seed 0 --log logs/erm/Terra_100 &
CUDA_VISIBLE_DEVICES=0 python erm.py data/Terra -d Terra -s 38 43 100 -t 38 -a resnet50 --seed 0 --log logs/erm/Terra_38 &
CUDA_VISIBLE_DEVICES=1 python erm.py data/Terra -d Terra -s 38 100 46 -t 43 -a resnet50 --seed 0 --log logs/erm/Terra_43 &
CUDA_VISIBLE_DEVICES=1 python erm.py data/Terra -d Terra -s 38 43 100 -t 46 -a resnet50 --seed 0 --log logs/erm/Terra_46
