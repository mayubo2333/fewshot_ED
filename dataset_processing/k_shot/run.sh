case "$1" in
  ACE)
    {  
        python construct_label_dict.py --dataset_type ACE
        python construct_fewshot_dataset.py --dataset_type ACE --train_K 2 --dev_K 1
        python construct_fewshot_dataset.py --dataset_type ACE --train_K 5 --dev_K 2
        python construct_fewshot_dataset.py --dataset_type ACE --train_K 10 --dev_K 2
    };;
  MAVEN)
    {  
        python construct_label_dict.py --dataset_type MAVEN
        python construct_fewshot_dataset.py --dataset_type MAVEN --train_K 2 --dev_K 1
        python construct_fewshot_dataset.py --dataset_type MAVEN --train_K 5 --dev_K 2
        python construct_fewshot_dataset.py --dataset_type MAVEN --train_K 10 --dev_K 2
    };;
  ERE)
    {  
        python construct_label_dict.py --dataset_type ERE
        python construct_fewshot_dataset.py --dataset_type ERE --train_K 2 --dev_K 1
        python construct_fewshot_dataset.py --dataset_type ERE --train_K 5 --dev_K 2
        python construct_fewshot_dataset.py --dataset_type ERE --train_K 10 --dev_K 2
    };;
  *)
    echo "Unknown dataset.";;
esac