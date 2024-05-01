# Workshop Comptext 2024 - Introduction to Text Analysis in R

This repository summarizes the Workshop "Introduction to Text Analysis in R" which took place the 6th Annual COMPTEXT at the Vrije Universiteit Amsterdam. 
The workshop introduced the attendees to basic concepts that are relevent to analyze text quantitatively using the programming language R.
Attendees were also given a practical overview of some of the introductory models and tools. 
The examples shown are a shorter summary of resources provided by the Department of [Communication Science at the VU Amsterdam](https://github.com/ccs-amsterdam) and the book [Computational Analysis of Communication](https://cssbook.net) by Van Atteveldt, Trilling and Arcila (2022). 

- [Basics in R](https://github.com/deniseroth97/workshop_comptext2024/blob/main/README.md#basics-in-r)

# Basics in R

## What are functions in R? 

A function in R is a block of code that performs a specific task when called. They typical take input parameters, perform operations on them, and then return a result based on these operations. They encapsulate commonly used operations into a single unit, making one's own work more transparent and reproducible. 
R already comes with a lot of built-in functions. 

Let's look at an example:


```{r Example of built-in function}
# Creating a vector of numbers
numbers <- c(2, 4, 6, 8, 10)

# Calculate the mean of the numbers using the mean()
average <- mean(numbers)

# Print the result
print(average)
```

The flexibility of R allows us to also write our own functions for very specific tasks and operations. 

What could this look?

```{r Writing a function}
# Defining the function
addition <- function(x, y) {
  sum <- x + y
  return(sum)
}

# Call the function with arguments 3 and 5
result <- addition(3, 5)

# Print the result
print(result)
```

## What is the meaning of the "->"-operator?

In R, the "->" symbol is used for assigning values to variables. It's like saying "put this value into that box." So when you see something like x -> y, it means the value of x is stored in the variable y.

R is an _object-oriented_ programming language - in simple terms, it means that everything in R is treated as an object. An object is like a container that holds data and functions that can work with that data. So when you're working in R, you're usually creating, manipulating, or using objects.

For example, when you create a variable like x <- 5, x is an object that holds the value 5. And when you perform operations or functions on x, you're essentially interacting with that object.

In our environment, we can see the objects to which we have assigned any sort of value etc.


```{r Objects}
random_object <- 27

print(random_object)
```

## Packages in R

In R, a package is like a toolbox filled with useful tools for specific tasks. Each package contains a collection of functions, datasets, and other resources designed to help you tackle certain problems more efficiently.

Packages make it easier to extend R's capabilities beyond its built-in functions. They're created and maintained by members of the R community, and you can easily install them using the ```install.packages()``` function.


```{r Installing packages}
install.packages("tidyverse")
```


Let's try using some functions that is found in the ```tidyverse``` package!


```{r}

# Defining an example data frame
example_data <- tibble(
  name = c("Alice", "Bob", "Charlie"),
  age = c(25, 30, 35)
)

# Attemptong to filtering data and keeping only the cases older than 30
plus_30s <- example_data |> 
  filter(age > 30)
```

*Important*: If we want to use a package and its functions/datasets/resources, we need to use the ```library()``` function to load the package. Otherwise, R will not know what we are referring to. 


# Text Preprocessing

## First things first: Setting up by installing all necessitities

We will now have a closer look at what it actually means to pre-process our textual data. Jumping from a full set of documents, what we often call a _corpus_ to a Document-Term Matrix (DTM) and using this immediately for analysis would be rather crude. 
Normally, there are several steps that come prior to this.
Let's have a look. 


```{r Install required packages}
install.packages(c("tidyverse","quanteda", "quanteda.textplots", "quanteda.textstats"))
library(quanteda)
library(quanteda.textplots)
library(quanteda.textstats)
library(tidyverse)
```


## Loading in the data

Let's use the State of the Union speeches as our corpus. Each document is (i.e. row in the csv) consists of a paragraph. The ```csv``` file we are using here is available online. We use the ```read_csv()``` to read in the data to R and assign it to an object. This way, our _dataframe_ will show up in our environment. 

```{r}
url <- 'https://bit.ly/2QoqUQS'
data <- read_csv(url)
head(data)
```


## Creating a corpus

Even though dataframes are nice to work with as they are very structure, the ```quanteda``` requires our textual data to be in the shape of a corpus object. In order to do so, we must specify the text column in our dataframe and this column must be a _character_ column, which is typical and usually preferably for a column containing any sort of text.


```{r}
sotu_corp <- corpus(data, text_field = 'text')  ## create the corpus
sotu_corp
```



## Creating DTM
Now that we have our corpus, we should think about whether all of the textual is actually useful for subsequent analysis. 
Can you think of any features (tokens) that are more or less informative?
*Tokens* are the individual units of text data, such as words or characters, that have been extracted or identified from the original text. In text analysis, text data is typically broken down into tokens for further processing and analysis.
Tokens are the basic building blocks used for representing and analyzing text data in various text mining and natural language processing tasks


```{r}
dtm <- sotu_corp |>
  tokens(remove_punct = T, remove_numbers = T, remove_symbols = T) |>   
  tokens_tolower() |>                                                    
  tokens_remove(stopwords('en')) |>                                     
  tokens_wordstem() |>
  dfm()
dtm
```

What do you think happens in the code above?

## Inspection of our Document-Term Matrix

```{r}
textplot_wordcloud(dtm, max_words = 50)                          ## this will plot the 50 most frequent words
textplot_wordcloud(dtm, max_words = 50, color = c('blue','red')) ## change colors; you can also try out different colors
textstat_frequency(dtm, n = 10)  
```


## How are and have US Presidents been talking about specific issues?
We can use keyword-in-context listing to search a specific term. Let's create a DTM  that only contains words that occur within 10 words from environment*.

```{r}
environment <- kwic(tokens(sotu_corp), 'environment*')
environment_corp <- corpus(environment)
environment_dtm <- environment_corp |>
  tokens(remove_punct = T, remove_numbers = T, remove_symbols = T) |>
  tokens_tolower() |>
  tokens_remove(stopwords('en')) |>
  tokens_wordstem() |>
  dfm()
```


Let's visualize this:

```{r}
textplot_wordcloud(environment_dtm, max_words = 50) 
```

## Conclusion

Which specific steps one takes to pre-process the textual data in question is strongly dependent on the research project one is involved in.
Keep in mind that there is no "standard" rule. There may instances where punctuation is highly informative. In other cases, someone may specifically be interested in whether certain terms are being capitalized or not. The choices we make at this stage of the process will have a significant impact on anything that follows. 


# Topic Models


## Setting Up

As is the case any time we want to use R, we must specify which and potentially install the packages that are necessary to conduct the analysis we envision for our data. For running LDA topic models, we need the ```topicmodels``` package. 


```{r}
install.packages(c("topicmodels", "LDAvis"))
library(quanteda)
library(quanteda.textplots)
library(LDAvis)
library(topicmodels)
library(wordcloud)
```


## Re-Using of our DTM
For this exercise, we will still use the DTM we created based on the corpus of SOTU addresses by US President until 2014. We will only make one additional change: We will remove features that appear fewer than 5 times. 

```{r}
dtm2 <- dfm_trim(dtm, min_docfreq = 5)
```


## Running an LDA Topic Model 

We now have convert our DTM to topicmodels format in order for the model to be able to run. Additionally, it is very important to set a seed before running topicmodels. This will make sure that other your research can be reproduced!

```{r}
tm_dtm = convert(dtm2, to = "topicmodels") 
set.seed(1)
model = LDA(tm_dtm, method = "Gibbs", k = 10,  control = list(alpha = 0.1))
model
```

## Inspection 

We can ask R to give us the top 10 terms per model. 

```{r}
terms(model, 10)
```


# Lexical Sentiment Analysis


## Setting Up

Let's install and load the necessary packages for this exercise. There are several packages in R that contain sentiment dictionaries.
For this exercise, we will have a closer look at the ```SentimentAnalysis``` package, and specifically at the ```DictionaryGI```, which is a general sentiment dictionary based on The General Inquirer.


```{r}
install.packages(c("corpustools", "SentimentAnalysis"))
library(quanteda)
library(quanteda.textplots)
library(quanteda.textstats)
library(corpustools)
library(SentimentAnalysis)
library(tidyverse)
```

## Creating the DTM and Taking Preprocessing Steps
We will again use the State of the Union addresses, which are incorporated in the Quanteda package. 


```{r}
corp <- corpus(sotu_texts, docid_field = 'id', text_field = 'text')
dtm <- corp |>
  tokens() |>
  tokens_tolower() |>
  dfm()
dtm
```



## The General Inquirer Dictionary

Let's inspect some of the words in our dictionary that indicate negative sentiment. 

```{r}
?DictionaryGI
names(DictionaryGI)
head(DictionaryGI$negative, 27)
```

## Transforming the DictionaryGI list to a quanteda dictionary

In order to be able to use quanteda functions, we must transform the list to an object that is useable for us.

```{r}
dict_GI <- dictionary(DictionaryGI)
```

```{r}
result <- dtm |>
  dfm_lookup(dict_GI) |> 
  convert(to = "data.frame") |>
  as_tibble()
result
```

## Adding the word count 

```{r}
result <- result |> 
  mutate(length = ntoken(dtm))
```


## Sentiment Score 
Let's consider calculating an overall sentiment score. There are several options we can explore, but a popular approach involves subtracting the count of negative sentiments from the count of positive sentiments. We then divide this by either the total number of words or by the number of sentiment words. Additionally, we can gauge the level of subjectivity expressed to understand the extent of sentiment conveyed overall.


```{r}
result <- result |> 
  mutate(sentiment1=(positive - negative) / (positive + negative),
         sentiment2=(positive - negative) / length,
         subjectivity=(positive + negative) / length)
result
```

These scores serve as indicators of sentiment and subjectivity on a per-document basis. To conduct a meaningful analysis, we coul merge these scores back into our metadata or document variables. This allows for the computation of sentiment metrics based on various attributes such as the document's source, actor, or temporal characteristics.


## Validation, validation, validation!


For a comprehensive assessment of a sentiment dictionary's accuracy, it's advisable to manually annotate a random subset of documents and juxtapose these annotations against the dictionary-derived results. This methodology should be clearly outlined in the methods section of any research paper employing a sentiment dictionary.

To generate a random subset from the original dataset, we can utilize the sample function.

```{r}
sample_ids <- sample(docnames(dtm), size=50)
```

```{r}
## convert quanteda corpus to data.frame
docs <- docvars(corp)
docs$doc_id = docnames(corp)
docs$text = as.character(corp)

docs |> 
  filter(doc_id %in% sample_ids) |> 
  mutate(manual_sentiment="") |>
  write_csv("to_code.csv")
```

Following this, you can export the sampled data to Excel, where you can manually assign sentiment labels to each document by populating the sentiment column. Subsequently, you can import the annotated data back into your analysis environment and merge it with the previously obtained results. It's worth mentioning that I've renamed the columns and converted the document identifier into a character column to streamline the matching process.


```{r}
validation = read_csv("to_code.csv") |>
  mutate(doc_id=as.character(doc_id)) |>
  inner_join(result)

cor.test(validation$manual_sentiment, validation$sentiment1)

validation <- validation |> 
  mutate(sent_nom = cut(sentiment1, breaks=c(-1, -0.1, 0.1, 1), labels=c("-", "0", "+")))
cm <- table(manual = vali3dation$manual_sentiment, dictionary = validation$sent_nom)
cm

sum(diag(cm)) / sum(cm)
```

We have the option to conduct a correlation analysis between the manually annotated sentiment labels and the dictionary-derived sentiment scores. Additionally, by converting the sentiment scores into a nominal value using the cut function, we can generate a "confusion matrix" to further evaluate the agreement between the manual annotations and the dictionary results.


## Creating one's own dictionary

Sentiment analysis frequently relies on a dictionary-based methodology, where predefined word lists or dictionaries are utilized to assess sentiment in textual data. It's worth noting that analysts have the option to develop their own dictionaries, offering a more customized approach tailored to their specific needs or research goals.



# Supervised Machine Learning for Automatic Text Classification

## Setting up 
```{r}
install.packages("tidymodels")
```


## Getting Data 

To train machine learning models, annotated training data is essential. Luckily, numerous review datasets are accessible for free. In this exercise, we'll utilize a collection of Amazon movie reviews stored as CSV files on our GitHub repository. For additional Amazon product reviews, including more recent ones, you can explore http://deepyeti.ucsd.edu/jianmo/amazon/index.html.


```{r}
library(tidyverse)
reviews = read_csv("https://raw.githubusercontent.com/ccs-amsterdam/r-course-material/master/data/reviews2.csv")
head(reviews)
table(reviews$overall)
```


Before we begin, let's establish a binary rating system based on the numerical overall rating.

```{r}
reviews = mutate(reviews, rating=as.factor(ifelse(overall==5, "good", "bad")))
```

The objective of this tutorial is to conduct supervised sentiment analysis, specifically, to predict the star rating based on the review text.


## Creating corpus

To commence, we'll utilize ```quanteda``` for textual preprocessing and ```quanteda.textmodels``` for running the machine learning algorithms.

Initially, we'll construct a corpus incorporating both the summary and review text. Additionally, we'll establish a binary rating system alongside the numeric overall score.

```{r}
library(quanteda)
review_corpus = reviews |> 
  mutate(text = paste(summary, reviewText, sep="\n\n"),
         doc_id = paste(asin, reviewerID, sep=":")) |>
  select(-summary, -reviewText) |>
  corpus()
```

Next, we'll divide the corpus into training and testing sets, ensuring reproducibility by setting the seed.

```{r}
set.seed(1)
testset = sample(docnames(review_corpus), 2000)
reviews_test =  corpus_subset(review_corpus, docnames(review_corpus) %in% testset)
reviews_train = corpus_subset(review_corpus, !docnames(review_corpus) %in% testset)
```

## Model Training

Initially, we'll generate a Document-Feature Matrix (DFM) from the training dataset, implementing stemming and trimming.

```{r}
dfm_train = reviews_train |> 
  tokens() |>
  dfm() |> 
  dfm_wordstem(language='english') |>
  dfm_select(min_nchar = 2) |>
  dfm_trim(min_docfreq=10)
```

Now, we can proceed to train a model, such as a Naive Bayes classifier.

```{r}
library(quanteda.textmodels)
rating_train = docvars(reviews_train, field="rating")
m_nb <- textmodel_nb(dfm_train, rating_train)
```

We can further examine the Naive Bayes (NB) parameters to identify the most significant predictors. These parameters represent the 'class conditional posterior estimates' (P(w|C)), but our interest lies in the opposite direction. By applying Bayes’ theorem (P(C|w) = P(w|C) * P(C) / P(w)), and assuming an equal prior distribution of classes, we can simplify to P(good|w) = P(w|good) / (P(w|good) + P(w|bad)).

```{r}
scores = t(m_nb$param) |> 
  as_tibble(rownames = "word") |> 
  mutate(relfreq=bad+good,
         bad=bad/relfreq,
         good=good/relfreq) 
scores |>
  filter(relfreq > .0001) |>
  arrange(-bad) |>
  head()
```

## Model Testing

To evaluate the model's performance, we apply it to the test data. It's crucial that the test data incorporates the same features (vocabulary) as the training data. The model comprises parameters for these features, not for words exclusive to the test data. Therefore, we skip the selection and trimming steps and instead adjust the test vocabulary (DFM columns) to match those in the training set.

```{r}
dfm_test = reviews_test |> 
  tokens() |>
  dfm() |> 
  dfm_wordstem(language='english') |>
  dfm_match(featnames(dfm_train))
```

Subsequently, we can classify the test data into their respective classes.

```{r}
nb_pred <- predict(m_nb, newdata = dfm_test)
head(nb_pred)
```

What can we see?

```{r}
predictions = docvars(reviews_test) |>
  as_tibble() |>
  add_column(prediction=nb_pred)
predictions |> 
  mutate(correct=prediction == rating) |>
  summarize(accuracy=mean(correct))
```


```{r}
library(tidymodels)
metrics = metric_set(accuracy, precision, recall, f_meas)
metrics(predictions, truth = rating, estimate = prediction)
conf_mat(predictions, truth = rating, estimate = prediction)
```




