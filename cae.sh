#!/usr/bin/env bash
# vitb16, PACS
CUDA_VISIBLE_DEVICES=0 python cae.py data/PACS -d PACS -t P -a vitb16 --seed 0 --log logs/cae/PACS_P &
CUDA_VISIBLE_DEVICES=0 python cae.py data/PACS -d PACS -t A -a vitb16 --seed 0 --log logs/cae/PACS_A &
CUDA_VISIBLE_DEVICES=1 python cae.py data/PACS -d PACS -t C -a vitb16 --seed 0 --log logs/cae/PACS_C &
CUDA_VISIBLE_DEVICES=1 python cae.py data/PACS -d PACS -t S -a vitb16 --seed 0 --log logs/cae/PACS_S

# vitb16, Office-Home
CUDA_VISIBLE_DEVICES=0 python cae.py data/office-home -d OfficeHome -t Pr -a vitb16 --seed 0 --log logs/cae/OfficeHome_Pr &
CUDA_VISIBLE_DEVICES=0 python cae.py data/office-home -d OfficeHome -t Rw -a vitb16 --seed 0 --log logs/cae/OfficeHome_Rw &
CUDA_VISIBLE_DEVICES=1 python cae.py data/office-home -d OfficeHome -t Cl -a vitb16 --seed 0 --log logs/cae/OfficeHome_Cl &
CUDA_VISIBLE_DEVICES=1 python cae.py data/office-home -d OfficeHome -t Ar -a vitb16 --seed 0 --log logs/cae/OfficeHome_Ar

# vitb16, DomainNet
CUDA_VISIBLE_DEVICES=0 python cae.py data/domainnet -d DomainNet -t c -a vitb16 --seed 0 --log logs/cae/DomainNet_c &
CUDA_VISIBLE_DEVICES=0 python cae.py data/domainnet -d DomainNet -t i -a vitb16 --seed 0 --log logs/cae/DomainNet_i &
CUDA_VISIBLE_DEVICES=0 python cae.py data/domainnet -d DomainNet -t p -a vitb16 --seed 0 --log logs/cae/DomainNet_p &
CUDA_VISIBLE_DEVICES=1 python cae.py data/domainnet -d DomainNet -t q -a vitb16 --seed 0 --log logs/cae/DomainNet_q &
CUDA_VISIBLE_DEVICES=1 python cae.py data/domainnet -d DomainNet -t r -a vitb16 --seed 0 --log logs/cae/DomainNet_r &
CUDA_VISIBLE_DEVICES=1 python cae.py data/domainnet -d DomainNet -t s -a vitb16 --seed 0 --log logs/cae/DomainNet_s

# vitb16, VLCS
CUDA_VISIBLE_DEVICES=0 python cae.py data/VLCS -d VLCS -t C -a vitb16 --seed 0 --log logs/cae/VLCS_C &
CUDA_VISIBLE_DEVICES=0 python cae.py data/VLCS -d VLCS -t L -a vitb16 --seed 0 --log logs/cae/VLCS_L &
CUDA_VISIBLE_DEVICES=1 python cae.py data/VLCS -d VLCS -t S -a vitb16 --seed 0 --log logs/cae/VLCS_S &
CUDA_VISIBLE_DEVICES=1 python cae.py data/VLCS -d VLCS -t V -a vitb16 --seed 0 --log logs/cae/VLCS_V

# vitb16, Terra
CUDA_VISIBLE_DEVICES=0 python cae.py data/Terra -d Terra -t 100 -a vitb16 --seed 0 --log logs/cae/Terra_100 &
CUDA_VISIBLE_DEVICES=0 python cae.py data/Terra -d Terra -t 38 -a vitb16 --seed 0 --log logs/cae/Terra_38 &
CUDA_VISIBLE_DEVICES=1 python cae.py data/Terra -d Terra -t 43 -a vitb16 --seed 0 --log logs/cae/Terra_43 &
CUDA_VISIBLE_DEVICES=1 python cae.py data/Terra -d Terra -t 46 -a vitb16 --seed 0 --log logs/cae/Terra_46
