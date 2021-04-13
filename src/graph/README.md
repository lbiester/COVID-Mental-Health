## User Graph Metrics

#### Basic metric computation logic
1. For each day's collection of Reddit posts/comments, form a graph where each unique user who made a post
   or comment is a node. Connect users who have "interacted" with each other (i.e. one user comments on another user's 
   post or comment) with an edge. Both directed (commenter -> commentee) and undirected versions of the graph are created.
2. Metrics are attributes of these graphs (density, diameter, etc.).


#### Instructions for computing metrics

1. **Collect graph metrics**. To collect graph metrics, run the following command:
    ```
    python src/graph/user_graph.py --subreddit_file <file containing list of subreddits you want to use in forming the user graph> \
                                   --start_date <first day to collect metrics for - MM/DD/YYYY format> \
                                   --end_date <last day (inclusive) to collect metrics for - MM/DD/YYYY format> \
                                   --output_dir <directory to output graph metrics to> \
                                   --graph_name <prefix to use in naming output files> \
                                   --num_groups <# of groups to split output into (helped with speeding up code) - default is 10 which should be fine>
                                  
    ```
   Note: I collected graph metrics from each subreddit individually, so when I ran the script I provided a subreddit_file that
   contained the name of a single subreddit.
   
2. **Process, filter, and aggregate metrics**. This step will aggregate and process the metrics output in the step above
to produce a single graph metrics CSV file that contains just the metrics relevant to our analysis and that is ready
to be input into the Prophet modeling step. To complete this step, run the following command:
    ```
   python src/graph/process_and_aggregate_graph_features.py --feature_dir <directory that contains the outputs from step #1> \
                                                            --output_file_name <prefix to use when naming output CSV file> \
                                                            --file_name_fmt <format string representing the names of the output of step #1, but with {} where the group numbers are listed> \
                                                            --num_groups <number of groups from step #1>
    ```
    Example for the `--file_name_fmt` argument: if the `graph_name` given in step #1 is "test", than the `file_name_fmt`
    given should be `group_{}_test.pkl`.
    
    After this step, there will be a file within the `feature_dir` named `{output_file_name}.csv`. This file is
    ready to be input to the Prophet model analysis.
