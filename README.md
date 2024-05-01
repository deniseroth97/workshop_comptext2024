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


