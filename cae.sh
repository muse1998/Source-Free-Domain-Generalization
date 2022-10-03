#!/usr/bin/env bash
# ResNet50, PACS
CUDA_VISIBLE_DEVICES=0 python cae.py data/PACS -d PACS -s A C S -t P -a vitb16 --freeze-bn --seed 0 --log logs/cae/PACS_P &
CUDA_VISIBLE_DEVICES=0 python cae.py data/PACS -d PACS -s P C S -t A -a vitb16 --freeze-bn --seed 0 --log logs/cae/PACS_A &
CUDA_VISIBLE_DEVICES=1 python cae.py data/PACS -d PACS -s P A S -t C -a vitb16 --freeze-bn --seed 0 --log logs/cae/PACS_C &
CUDA_VISIBLE_DEVICES=1 python cae.py data/PACS -d PACS -s P A C -t S -a vitb16 --freeze-bn --seed 0 --log logs/cae/PACS_S

# ResNet50, Office-Home
CUDA_VISIBLE_DEVICES=0 python cae.py data/office-home -d OfficeHome -s Ar Cl Rw -t Pr -a vitb16 --seed 0 --log logs/cae/OfficeHome_Pr &
CUDA_VISIBLE_DEVICES=0 python cae.py data/office-home -d OfficeHome -s Ar Cl Pr -t Rw -a vitb16 --seed 0 --log logs/cae/OfficeHome_Rw &
CUDA_VISIBLE_DEVICES=1 python cae.py data/office-home -d OfficeHome -s Ar Rw Pr -t Cl -a vitb16 --seed 0 --log logs/cae/OfficeHome_Cl &
CUDA_VISIBLE_DEVICES=1 python cae.py data/office-home -d OfficeHome -s Cl Rw Pr -t Ar -a vitb16 --seed 0 --log logs/cae/OfficeHome_Ar

# ResNet50, DomainNet
CUDA_VISIBLE_DEVICES=0 python cae.py data/domainnet -d DomainNet -s i p q r s -t c -a vitb16 --seed 0 --log logs/cae/DomainNet_c &
CUDA_VISIBLE_DEVICES=0 python cae.py data/domainnet -d DomainNet -s c p q r s -t i -a vitb16 --seed 0 --log logs/cae/DomainNet_i &
CUDA_VISIBLE_DEVICES=0 python cae.py data/domainnet -d DomainNet -s c i q r s -t p -a vitb16 --seed 0 --log logs/cae/DomainNet_p &
CUDA_VISIBLE_DEVICES=1 python cae.py data/domainnet -d DomainNet -s c i p r s -t q -a vitb16 --seed 0 --log logs/cae/DomainNet_q &
CUDA_VISIBLE_DEVICES=1 python cae.py data/domainnet -d DomainNet -s c i p q s -t r -a vitb16 --seed 0 --log logs/cae/DomainNet_r &
CUDA_VISIBLE_DEVICES=1 python cae.py data/domainnet -d DomainNet -s c i p q r -t s -a vitb16 --seed 0 --log logs/cae/DomainNet_s

# ResNet50, VLCS
CUDA_VISIBLE_DEVICES=0 python cae.py data/VLCS -d VLCS -s La Su Vo -t Ca -a vitb16 --seed 0 --log logs/cae/VLCS_Ca &
CUDA_VISIBLE_DEVICES=0 python cae.py data/VLCS -d VLCS -s La Su Vo -t La -a vitb16 --seed 0 --log logs/cae/VLCS_La &
CUDA_VISIBLE_DEVICES=1 python cae.py data/VLCS -d VLCS -s La Su Vo -t Su -a vitb16 --seed 0 --log logs/cae/VLCS_Su &
CUDA_VISIBLE_DEVICES=1 python cae.py data/VLCS -d VLCS -s La Su Vo -t Vo -a vitb16 --seed 0 --log logs/cae/VLCS_Vo

# ResNet50, Terra
CUDA_VISIBLE_DEVICES=0 python cae.py data/Terra -d Terra -s 38 43 46 -t 100 -a vitb16 --seed 0 --log logs/cae/Terra_100 &
CUDA_VISIBLE_DEVICES=0 python cae.py data/Terra -d Terra -s 38 43 100 -t 38 -a vitb16 --seed 0 --log logs/cae/Terra_38 &
CUDA_VISIBLE_DEVICES=1 python cae.py data/Terra -d Terra -s 38 100 46 -t 43 -a vitb16 --seed 0 --log logs/cae/Terra_43 &
CUDA_VISIBLE_DEVICES=1 python cae.py data/Terra -d Terra -s 38 43 100 -t 46 -a vitb16 --seed 0 --log logs/cae/Terra_46
