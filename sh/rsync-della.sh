mkdir -p /scratch/tmp/sa7998/clm-input/

# transfer clm inputs from della
rsync -avzP --ignore-existing \
  della:/home/sa7998/git/biosphere-metabolome/data/clm-input/ \
  /scratch/tmp/sa7998/clm-input/
