## Instructions for Using a Trained Topic Model to Extract Topic Metrics

1. **Locate the files associated with the trained topic model you want to use.** You need both the file storing the trained model (a .mdl file) and a file that has the vocabulary associated with the 
 model. For the topic model used in our ACL workshop paper, these files are located in the 
 `data/topic_model/dep_anx_sw_posts_only_17_20 directory`. The files you need are `dep_anx_sw_posts_only_25_topics.mdl`
  and `dep_anx_sw_posts_vocab.dct`.
  
    One tricky thing about the [Gensim mallet topic model](https://radimrehurek.com/gensim/models/wrappers/ldamallet.html)
    that I used is that when you intialize the model, you can specify a 'prefix' for 'produced temporary files'. However,
    what's confusing is that these files aren't really 'temporary' - instead, they are needed if you want to use any saved
    model in the future. The 'temporary' files associated with the topic model used in our project are located in the
    `/data/topic_model/mallet_tmp_files/` directory. Check to make sure you have these before moving on to the next steps!
  
2. **Collect and preprocess the documents you want to run the topic model on.** To do this, run the following command: 
    ```
    python src/topics/corpus.py --subreddit_file <path to file containing list of subreddits you want to use> \
                                --name <name of corpus/doc collection to use in output file names> \
                                --start_date <start date for collection of posts - MM/DD/YYYY format> \
                                --end_date <end date for collection of posts (exclusive) - MM/DD/YYYY format> \
                                --output_dir <path to directory to output files to> \
                                --vocab data/topic_model/dep_anx_sw_posts_only_17_20/dep_anx_sw_posts_vocab.dct
    ```
    The above command is specific to using the pre-trained model from our paper. If you want to use a different model,
    specify the path to its vocabulary file as the `--vocab` argument. Note that the code is currently setup to just extract subreddit posts (not comments). If you want to add collection of
    comments, this can be done pretty easily - you just need to add it as a command line argument and update how the Corpus class
    is initialized.
    
    The above script also contains a `--downsample` optional argument. If provided, a corpus will be formed such that the data 
    is downsampled to ensure that there are an equal number of posts collected from each day, and for each day, there are an equal
    number of posts from each subreddit. This may be useful if you want to collect a corpus of documents from the baseline
    subreddits such that each individual subreddit is equally represented.
    
3. **Compute topic usage metrics from your corpus of documents (from step 2) using the pre-trained model (from step 1).**
    To do this, run the following command:
    ```
    python src/topics/get_document_topic_distributions.py --topic_model_path data/topic_model/dep_anx_sw_posts_only_17_20/dep_anx_sw_posts_only_25_topics.mdl \
                                                          --corpus_dir <directory with corpus of docs to run topic model on - i.e. 'output dir' from step #2> \
                                                          --corpus_name <prefix used in naming corpus files - i.e. 'name' from step #2> \
                                                          --output_dir <path to directory to save topic distribution metrics to> \
                                                          --vocab_path data/topic_model/dep_anx_sw_posts_only_17_20/dep_anx_sw_posts_vocab.dct \
    ```
    This script aggregates all posts in the  input corpus (i.e. corpus from step #2) for each day and computes topic usage metrics for each day. With this implementation,
    this means that if you want to extract subreddit-specific topic usage metrics, you need to create a corpus for each subreddit in step 2, and then
    run the above command for each subreddit's corpus. If this becomes too cumbersome, you should be able to adapt the script above
    relatively easily to have it filter posts from the input corpus by subreddit.
   
    Note that the above command is specific to using the pre-trained model from our paper. If you don't want to use this model,
    you'll need to update the `--topic_model_path` and `vocab_path` arguments accordingly. 
    
    After this step, you should have a file called <corpus_name>_<post_type>_topic_dist.csv, which can be used as input into the 
    Prophet model analysis.