case "$1" in
  ACE)
    {  
        python construct_fewshot_dataset.py --dataset_type ACE --train_K 2 --dev_K 1
        python construct_fewshot_dataset.py --dataset_type ACE --train_K 5 --dev_K 2 --only_few_shot
        python construct_fewshot_dataset.py --dataset_type ACE --train_K 10 --dev_K 2 --only_few_shot
    };;
  MAVEN)
    {  
        python construct_fewshot_dataset.py --dataset_type MAVEN --train_K 2 --dev_K 1
        python construct_fewshot_dataset.py --dataset_type MAVEN --train_K 5 --dev_K 2 --only_few_shot
        python construct_fewshot_dataset.py --dataset_type MAVEN --train_K 10 --dev_K 2 --only_few_shot
    };;
  ERE)
    {  
        python construct_fewshot_dataset.py --dataset_type ERE --train_K 2 --dev_K 1
        python construct_fewshot_dataset.py --dataset_type ERE --train_K 5 --dev_K 2 --only_few_shot
        python construct_fewshot_dataset.py --dataset_type ERE --train_K 10 --dev_K 2 --only_few_shot
    };;
  *)
    echo "Unknown dataset.";;
esac