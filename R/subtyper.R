# Here's a good place to put your top-level package documentation

.onLoad <- function (lib, pkgname="subtyper") {
    ## Put stuff here you want to run when your package is loaded
    invisible()
}


#' Extract Complete Cases from a Data Frame Based on a Model Equation
#'
#' This function takes a data frame and a model equation as input, and returns
#' a boolean vector indicating which rows of the data frame have complete cases
#' for the variables specified in the equation.
#'
#' @param df A data frame containing the variables specified in the equation.
#' @param equation A model equation specifying the variables of interest.
#' @return A boolean vector indicating which rows of the data frame have complete cases.
#' @examples
#' df <- data.frame(
#'   x = c(1, 2, NA, 4),
#'   y = c(NA, 2, 3, 4),
#'   z = c(1, 2, 3, NA)
#' )
#' equation <- y ~ x + z
#' complete_index <- complete_cases_from_equation(df, equation)
#' print(complete_index)  # Output: FALSE TRUE FALSE FALSE
#' @export
complete_cases_from_equation <- function(df, equation) {
  # Extract variable names from the equation using all.vars
  vars <- all.vars(equation)
  
  # Subset the dataframe to include only the relevant variables
  relevant_data <- df[, vars, drop = FALSE]
  
  # Use complete.cases to get a boolean vector indicating complete cases
  complete_cases <- complete.cases(relevant_data)
  
  return(complete_cases)
}


#' Compare Model Predictions with True Values
#'
#' This function takes in two models and produces two ggplots in a list that show the predictions
#' of the models versus the true response variable. For mixed models, it allows the user to ignore random effects.
#' It also displays the R-squared (R²) value and the difference in R-squared (ΔR²) between the two models within each plot.
#'
#' @param model1 A fitted model object (either a standard regression or a mixed model).
#' @param model2 A second fitted model object (either a standard regression or a mixed model).
#' @param data A data frame containing the variables used in the models.
#' @param response The name of the response variable as a string.
#' @param re.form A formula specifying which random effects to include in predictions, or `NA` to exclude them. Defaults to including all random effects. Only applies if the model is a mixed model.
#' @param custom_title An optional vector of custom titles for the plots.
#' @param annot_size optional size for text annotation (default 5)
#' @return A list of two ggplot objects, each comparing the predictions of a model with the true response variable.
#' @import ggplot2
#' @importFrom stats predict
#' @importFrom methods is
#' @examples
#' \dontrun{
#'   mdl1 <- lm(Sepal.Length ~ Sepal.Width + Petal.Length, data = iris)
#'   mdl2 <- lm(Sepal.Length ~ Sepal.Width + Petal.Width, data = iris)
#'   plots <- compare_model_predictions(mdl1, mdl2, data = iris, response = "Sepal.Length")
#'   plots[[1]] # Plot for model 1
#'   plots[[2]] # Plot for model 2
#' }
#' @export
compare_model_predictions <- function(model1, model2, data, response, re.form = NULL, custom_title=NULL, annot_size=5 ) {
  # Ensure response is in the data
  if (!response %in% colnames(data)) {
    stop("The specified response variable is not in the data frame.")
  }
  
  # Determine if the models are mixed models
  is_mixed_model1 <- is(model1, "merMod") # lme4 models return class "merMod"
  is_mixed_model2 <- is(model2, "merMod")
  
  # Get predictions from model 1
  if (is_mixed_model1) {
    preds1 <- predict(model1, newdata = data, re.form = re.form)
    r_squared1 <- round( as.numeric(rsq( model1 )[2]), 3 )
  } else {
    preds1 <- predict(model1, newdata = data)
    r_squared1 <- round( as.numeric(rsq( model1 )[1]), 3 )
  }
  
  # Get predictions from model 2
  if (is_mixed_model2) {
    preds2 <- predict(model2, newdata = data, re.form = re.form)
    r_squared2 <- round( as.numeric(rsq( model2 )[2]), 3 )
  } else {
    preds2 <- predict(model2, newdata = data)
    r_squared2 <- round( as.numeric(rsq( model2 )[1]), 3 )
  }
  
  # Calculate delta R-squared
  delta_r_squared <- round(r_squared2 - r_squared1, 3)
  
  # Create data frame for plotting
  plot_data <- data.frame(
    True = data[[response]],
    Pred1 = preds1,
    Pred2 = preds2
  )
  
  if ( is.null( custom_title ) & FALSE ) {
    custom_title = c( 
      paste("Base Model vs True", response)  ,
      paste("Ext. Model vs True", response)  
    )
  }
  
  # Find the minimum and maximum values for the axis limits
  min_val <- min(c(plot_data$True, plot_data$Pred1, plot_data$Pred2))
  max_val <- max(c(plot_data$True, plot_data$Pred1, plot_data$Pred2))
  
  # Create plots with informative titles, axis labels, R-squared values, delta R-squared, and best-fit regression lines
  plot1 <- ggplot(plot_data, aes(x = True, y = Pred1)) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE, color = "blue", linetype = "solid") +  # Best-fit regression line
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(
      title = custom_title[1],
      x = paste("T.", response),
      y = paste("P.", response)
    ) +
    annotate(geom="label", x = max_val, y = min_val, label = paste("R² =", r_squared1, "\nΔR² =", delta_r_squared),
             hjust = 1.1, vjust = -0.5, size = annot_size, color = "blue", fill = 'white') +
    scale_x_continuous(limits = c(min_val, max_val)) +
    scale_y_continuous(limits = c(min_val, max_val)) +
    theme_minimal()
  
  plot2 <- ggplot(plot_data, aes(x = True, y = Pred2)) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE, color = "blue", linetype = "solid") +  # Best-fit regression line
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(
      title = custom_title[2],
      x = paste("T.", response),
      y = paste("P.", response)
    ) + 
    annotate(geom="label", x = max_val, y = min_val, label = paste("R² =", r_squared2, "\nΔR² =", delta_r_squared),
             hjust = 1.1, vjust = -0.5, size = annot_size, color = "blue", fill = 'white') +
    scale_x_continuous(limits = c(min_val, max_val)) +
    scale_y_continuous(limits = c(min_val, max_val)) +
    theme_minimal()
  
  # Return the plots in a list
  return(list(plot1, plot2))
}

#' Select Longitudinal Subjects
#'
#' Select subjects that have at least a minimum number of visits and
#' have occurrences at each instance of the time column. Optionally,
#' return only subjects where the time column equals a user-specified
#' baseline value.
#'
#' @param dataframe The input dataframe.
#' @param subjectID_column The column name of the subject IDs.
#' @param time_column The column name of the time variable.
#' @param min_visits The minimum number of visits required.
#' @param baseline Optional baseline value to filter by. If specified,
#'   only subjects with a time column value equal to this baseline will
#'   be returned.
#'
#' @return A dataframe containing the selected subjects.
#'
#' @examples
#' dataframe <- data.frame(
#'   subject_id = c(1, 1, 1, 2, 2, 3, 3, 3),
#'   time = c(0, 1, 2, 0, 1, 0, 1, 2),
#'   value = rnorm(8)
#' )
#'
#' # Select subjects with at least 3 visits and occurrences at each time point
#' selected_subjects <- select_longitudinal_subjects(dataframe, "subject_id", "time", 3)
#'
#' # Select subjects with at least 2 visits and occurrences at each time point,
#' # and return only subjects with a time column value equal to 0
#' selected_subjects <- select_longitudinal_subjects(dataframe, "subject_id", "time", 2, baseline = 0)
#' @export
select_longitudinal_subjects <- function(dataframe, subjectID_column, time_column, min_visits, baseline = NULL) {
  # Create a contingency table (cross-tabulation) of subjects by time points
  subject_time_table <- table(dataframe[[subjectID_column]], dataframe[[time_column]])
  
  # Initialize an empty vector to store the IDs of valid subjects
  valid_subjects <- c()
  
  # Get the expected number of time points (columns in the contingency table)
  expected_time_points <- ncol(subject_time_table)
  
  # Loop through each subject to check if they meet the criteria
  for (subject in rownames(subject_time_table)) {
    # Get the number of visits at each time point for the current subject
    visits_per_time_point <- subject_time_table[subject, ]
    
    # Check if the subject has visits at every time point
    has_all_time_points <- all(visits_per_time_point > 0)
    
    # Check if the subject has at least the minimum number of visits
    total_visits <- sum(visits_per_time_point)
    has_min_visits <- total_visits >= min_visits
    
    # If a baseline is specified, check if the subject has a visit at the baseline time point
    if (!is.null(baseline)) {
      has_baseline_visit <- visits_per_time_point[as.character(baseline)] > 0
    } else {
      has_baseline_visit <- TRUE
    }
    
    # If the subject meets all the criteria, add them to the list of valid subjects
    if ( has_min_visits && has_baseline_visit) {
      valid_subjects <- c(valid_subjects, subject)
    }
  }
  
  # Filter the original dataframe to return only the rows corresponding to the valid subjects
  selected_data <- dataframe[dataframe[[subjectID_column]] %in% valid_subjects, ]
  
  # Explicit check: Ensure that each selected subject has at least min_visits
  subject_visit_counts <- table(selected_data[[subjectID_column]])
  valid_subjects_final <- names(subject_visit_counts[subject_visit_counts >= min_visits])
  
  # Filter again to ensure only subjects with at least min_visits are returned
  selected_data <- selected_data[selected_data[[subjectID_column]] %in% valid_subjects_final, ]
  
  return(selected_data)
}


#' Select Top k Rows Based on a Criterion
#'
#' This function selects the top `k` rows of a dataframe according to a specified 
#' column and criterion (either maximizing or minimizing the values). It returns 
#' a logical vector indicating which rows are among the top `k`.
#'
#' @param df A dataframe from which to select the top rows.
#' @param column A string specifying the column name to base the selection on.
#' @param k An integer specifying the number of top rows to select.
#' @param maximize A logical value indicating whether to maximize (`TRUE`) or 
#' minimize (`FALSE`) the criterion. Default is `TRUE`.
#' 
#' @return A logical vector of the same length as the number of rows in `df`. 
#' `TRUE` indicates that the row is one of the top `k`.
#'
#' @examples
#' df <- data.frame(
#'   id = 1:10,
#'   value = c(5, 2, 9, 4, 7, 3, 6, 10, 8, 1)
#' )
#' 
#' # Select the top 3 rows based on maximizing the 'value' column
#' topk(df, "value", 3, maximize = TRUE)
#'
#' # Select the top 3 rows based on minimizing the 'value' column
#' topk(df, "value", 3, maximize = FALSE)
#'
#' @export
topk <- function(df, column, k, maximize = TRUE) {
  if (!(column %in% names(df))) {
    stop("The specified column does not exist in the dataframe.")
  }
  
  if (maximize) {
    ranking <- order(df[[column]], decreasing = TRUE)
  } else {
    ranking <- order(df[[column]], decreasing = FALSE)
  }
  
  selected_rows <- ranking[1:k]
  
  result <- rep(FALSE, nrow(df))
  result[selected_rows] <- TRUE
  
  return(result)
}

#' Filter Columns by Percentage of NA Values
#'
#' @description
#' This function filters out columns from a dataframe that have a percentage of
#' NA (missing) values greater or equal to a user-defined threshold. The percentage
#' is calculated with respect to the total number of rows in the dataframe.
#'
#' @param df A dataframe from which columns will be filtered.
#' @param percentThreshold A numeric value between 0 and 100 indicating the percentage
#'        threshold of NA values. Columns with NA percentages greater or equal to this
#'        value will be removed from the dataframe.
#'
#' @return A dataframe with columns filtered based on the NA percentage threshold.
#'
#' @examples
#' data(mtcars)
#' mtcars[1:5, 1:2] <- NA # Introduce NA values for demonstration
#' filtered_df <- filterNAcolumns(mtcars, 10)
#' print(filtered_df)
#'
#' @export
#'
#' @import dplyr
filterNAcolumns <- function(df, percentThreshold) {
  # Validate input
  if (!is.data.frame(df)) {
    stop("The input 'df' must be a dataframe.")
  }
  
  if (!is.numeric(percentThreshold) || percentThreshold < 0 || percentThreshold > 100) {
    stop("The 'percentThreshold' must be a numeric value between 0 and 100.")
  }
  
  # Calculate the percentage of NA values for each column
  na_percent <- sapply(df, function(col) {
    sum(is.na(col)) / nrow(df) * 100
  })
  
  # Filter columns based on threshold
  filtered_df <- df[, na_percent < (100.0-percentThreshold)]
  
  return(filtered_df)
}

#' Interpret ICC Value
#'
#' Interprets the Intra-Class Correlation Coefficient (ICC) value based on 
#' the criteria set by Cicchetti (1994) or Koo (2015).
#'
#' @param icc Numeric value; the ICC value to interpret.
#' @param criterion Character string; either "Cicchetti" or "Koo", 
#' indicating which set of criteria to use for interpretation. 
#' Defaults to "Cicchetti".
#'
#' @return A character string indicating the level of agreement or 
#' reliability as per the chosen criterion.
#'
#' @examples
#' interpret_icc(0.65, criterion = "Cicchetti")
#' interpret_icc(0.65, criterion = "Koo")
#'
#' @export
interpret_icc <- function(icc, criterion = c("Cicchetti", "Koo")) {
  criterion <- match.arg(criterion)
  
  # Cicchetti's criteria
  if (criterion == "Cicchetti") {
    if (icc < 0.40 & icc >= 0 ) {
      return("Slight")
    } else if (icc < 0.60 & icc >= 0) {
      return("Fair")
    } else if (icc < 0.75 & icc >= 0) {
      return("Moderate")
    } else if (icc <= 1.00 & icc >= 0 )  {
      return("Substantial")
    } else return(NA)
  } 
  # Koo's criteria
  else if (criterion == "Koo") {
    if (icc < 0.50 & icc >= 0 ) {
      return("Slight")
    } else if (icc < 0.75 & icc >= 0) {
      return("Fair")
    } else if (icc < 0.90 & icc >= 0 ) {
      return("Moderate")
    } else if (icc <= 1.00 & icc >= 0) {
      return("Substantial")
    } else return(NA)
  }
}


#' Convert nrg format date to R format date
#'
#' @param x nrg format date
#' @return fixed x
#' @author Avants BB
#' @export
nrgDateToRDate <- function( x ) {
  for ( k in 1:length(x)) {
    x[k]=paste0( substr(x[k],1,4), "-", substr(x[k],5,6)  , "-", substr(x[k],7,8))
  }
  as.Date( x, format="%Y-%m-%d" )
}


#' Minimize the difference between two data frames based on t-statistic.
#'
#' This function generates random subsets of a data frame to minimize the difference with another data frame based on a specified set of columns, as measured by the t-statistic.  Authored by Avants and Chat-GPT 3.5.
#'
#' @param df1 Data frame to be subsetted.
#' @param df2 Data frame used as a reference for comparison.
#' @param cols Vector of column names used for matching.
#' @param sample_size the number to sample from df1
#' @param num_iterations Number of random subsets to generate.
#' @param restrict_df1 float lower quantile to restrict df1 based on first col value to match range of df2
#' @param option either random or optimal
#' @param verbose boolean
#'
#' @return rownames of a sub data frame that minimizes the difference with df2 in terms of t-statistic.
#'
#' @examples
#' set.seed(123)
#' df1 <- data.frame(A = rnorm(100), B = factor(sample(1:3, 100, replace = TRUE)), C = rnorm(100))
#' df2 <- data.frame(A = rnorm(50), B = factor(sample(1:3, 50, replace = TRUE)), C = rnorm(50))
#' matching_cols <- c("A", "B")
#' # matched_subset <- match_cohort_pair(df1, df2, matching_cols)
#' # print(matched_subset)
#'
#' @importFrom stats t.test
#' @importFrom nmfbin nmfbin
#' @importFrom MASS ginv
#' @importFrom optmatch pairmatch match_on
#' @export
match_cohort_pair <- function(df1, df2, cols, sample_size, num_iterations = 1000, restrict_df1=0.05, option='optimal', verbose=TRUE ) {

  stopifnot(  all( cols %in% colnames(df1) ) )
  stopifnot(  all( cols %in% colnames(df2) ) )
  
  if ( restrict_df1 > 0) {
      df1=df1[  
          df1[,cols[1]] > quantile(df2[,cols[1]],restrict_df1,na.rm=T) &
          df1[,cols[1]] < quantile(df2[,cols[1]],1.0-restrict_df1,na.rm=T),  ]
  }

  if ( missing( sample_size ) ) sample_size = min( c(nrow(df2),nrow(df1)))
  if ( sample_size > nrow(df1 ) ) sample_size = nrow( df1 ) - 1
  if ( option != 'optimal' ) {
    best_subset <- NULL
    min_t_statistic <- Inf
    
    # Convert factor columns to character for t-test
    for (col in cols) {
      if (!is.numeric(df1[,col])) {
        df1[,col] <- as.numeric(as.factor(df1[,col]))
      }
      if (!is.numeric(df2[,col])) {
        df2[,col] <- as.numeric(as.factor(df2[,col]))
      }
    }
    t_statistic=rep(NA,length(cols))
    names(t_statistic)=cols
    min_t_statistic=rep(Inf,length(cols))
    names(min_t_statistic)=cols
    for (i in 1:num_iterations) {
      # Randomly subset df1 based on the columns
      subset_indices <- sample(1:nrow(df1), 
        size = sample_size, replace = FALSE )
      # print( head( subset_indices ) )
      subset_df1 <- df1[subset_indices, cols]
      if ( length( cols) == 1 ) {
        subset_df1=data.frame(df1[subset_indices, cols] )
        colnames(subset_df1)=cols
      } 
      # Calculate t-statistic for the subset
      for (col in cols) {
          t_statistic[col]=abs(t.test(subset_df1[,col], df2[,col])$statistic)
          }
      
      # Check if the current subset has a smaller t-statistic
      if ( mean(t_statistic) < mean(min_t_statistic) ) {
        min_t_statistic <- t_statistic
        best_subset <- subset_indices
        if ( verbose ) print( paste( min_t_statistic ))
      }
    }
    if ( verbose ) print( paste( min_t_statistic ))
    return( rownames(df1)[best_subset] )
  } else {
    df1$my_match_cohort_pair_var = 0
    df2$my_match_cohort_pair_var = 1
    temp = rbind( df1, df2 )
    distances <- list()
    myform = paste0( "my_match_cohort_pair_var ~",  paste(cols,collapse="+") )
    distances$mahal <- match_on( as.formula(myform), data = temp)
    mahal.match <- pairmatch(distances$mahal, data = temp, remove.unmatchables = TRUE )
    matchednames = names( mahal.match )
    return( matchednames[ matchednames %in% rownames(df1) ] )
    return(  list( 
      df1=matchednames[ matchednames %in% rownames(df1) ], 
      df2=matchednames[ matchednames %in% rownames(df2) ] ) )
  }
}


#' Convert left/right variables to a measure of asymmetry
#'
#' @param mydataframe dataframe containing relevant variables
#' @param leftvar left side variable names ie the full names of the variables to asym
#' @param leftname the variable substring indicating left side
#' @param rightname the variable substring indicating right side
#' @param replacer string to replace left with in column names of output
#' @return fixed x
#' @author Avants BB
#' @export
mapAsymVar <-function( mydataframe, leftvar, leftname='left', rightname='right', replacer='Asym' ) {

  library(stringr)
  library(purrr)
  replace_values <- function(input_string) {
    # Function to modify a number based on the specified rules
    modify_value <- function(number) {
      num <- as.numeric(number)
      if (num >= 1 && num <= 249) {
        return(as.character(num + 249))
      } else {
        return(number)
      }
    }
    
    # Extract all numbers from the string
    numbers <- str_extract_all(input_string, "\\b\\d+\\b")[[1]]
    
    # Apply the modification to the numbers
    modified_numbers <- map_chr(numbers, modify_value)
    
    # Replace old numbers with new numbers in the string
    for (i in seq_along(numbers)) {
      input_string <- str_replace(input_string, numbers[i], modified_numbers[i])
    }

    return(input_string)
  }

  rightvar =  gsub( leftname, rightname, leftvar )
#  for ( k in 1:length(rightvar) ) {
#    r=rightvar[k]
#    if ( length( grep("rsfMRI_",r) > 0 ) )
#      rightvar[k]=replace_values(r)
#  }
  hasright = rightvar %in% colnames(mydataframe)
  temp = mydataframe[,leftvar[hasright]] - mydataframe[,rightvar[hasright]]
  temp = temp * sign(temp )
  newnames = gsub(leftname, replacer,leftvar[hasright])
  mydataframe[,newnames]=temp
  return( mydataframe )
}



#' Convert left/right variables to an average measurement
#'
#' @param mydataframe dataframe containing relevant variables
#' @param leftvar left side variable names ie the full names of the variables to average
#' @param leftname the variable substring indicating left side
#' @param rightname the variable substring indicating right side
#' @param replacer string to replace left with in column names of output
#' @return fixed x
#' @author Avants BB
#' @export
mapLRAverageVar <- function( mydataframe, leftvar, leftname='left',rightname='right', replacer='LRAVG' ) {
  rightvar =  gsub( leftname, rightname, leftvar )
  hasright = rightvar %in% colnames(mydataframe)
  temp = mydataframe[,leftvar[hasright]] * 0.5 + mydataframe[,rightvar[hasright]] * 0.5
  newnames = gsub(leftname, replacer,leftvar[hasright])
  mydataframe[,newnames]=temp
  return( mydataframe )
}


#' Convert NA to false
#'
#' @param x a vector of bool
#' @return fixed x
#' @author Avants BB
#' @export
fs <- function( x ) {
  x[ is.na(x)]=FALSE
  x
}


#' Convert NA to false
#'
#' @param x a vector of bool
#' @return fixed x
#' @author Avants BB
#' @export
na2f <- function( x ) {
  x[ is.na(x)]=FALSE
  x
}

#' Grep entries with a vector search parameters
#'
#' @param x a vector of search terms
#' @param desc target vector of items to be searched
#' @param intersect boolean whether to use intersection or union otherwise
#'
#' @return result of grep
#' @author Avants BB
#' @export
multigrep <- function( x, desc, intersect=FALSE ) {
  roisel = c()
  for ( xx in x ) {
    if (length(roisel)==0 | !intersect ) {
      roisel = c( roisel, grep(xx, desc) )
    } else {
      roisel = intersect( roisel, grep(xx, desc) )
    }
  }
  return(  roisel )
}



#' Extract column names with concatenated search parameters
#'
#' @param x vector of strings
#' @param demogIn the dataframe with column names to search.
#' @param exclusions the strings to exclude
#'
#' @return vector of string column names
#' @author Avants BB
#' @examples
#'
#' mydf = generateSubtyperData( 5 )
#' nms = getNamesFromDataframe( c("it","v"), mydf )
#'
#' @export
getNamesFromDataframe <- function( x, demogIn, exclusions ) {
  outnames = names(demogIn)[ grep(x[1],names(demogIn ) ) ]
  if ( length( x ) > 1 )
  for ( y in x[-1] )
    outnames = outnames[ grep(y,outnames ) ]

  if ( ! missing( exclusions ) ) {
    toexclude=grep(exclusions[1],outnames)
    if ( length(exclusions) > 1 )
      for ( zz in exclusions[-1] ) {
        toexclude = c( toexclude, grep(zz,outnames) )
      }
    if ( length( toexclude ) > 0 ) outnames = outnames[ -toexclude ]
  }
  return( outnames )
}


#' nallspdma
#'
#' Supplementary table S2 from Nalls 2019 Parkinsons disease meta-analysis of
#' genome wide association studies.
#'
#' @docType data
#' @references \url{https://pubmed.ncbi.nlm.nih.gov/31701892/}
#' @format A data frame (subset of columns described here):
#' \describe{
#'   \item{SNP}{the SNP of interest}
#'   \item{CHR}{associated chromosome}
#'   \item{BP}{base pair}
#'   \item{NearestGene}{nearest gene}
#' }
#' @usage data(nallspdma)
"nallspdma"


#' Generate example data for illustrating `subtyper`
#'
#' @param n numer of subjects
#' @param groupmeansbytime a 9-vector for each group by time mean for cognition.
#' the cognitive data is simulated from these mean values.  Group 3 changes
#' more rapidly, by default.
#'
#' @return data frame
#' @author Avants BB
#' @examples
#'
#' mydf = generateSubtyperData( 5 )
#'
#' @export
generateSubtyperData <-function( n = 100,
  groupmeansbytime = c(
      1, 3, 8, # V0 G0,G1,G2
      0.95, 3.5, 12,  # V1 G0,G1,G2
      1.05, 4.0, 20 )  # V2 G0,G1,G2
) {

  myids = LETTERS[1:24]
  groupn = round( length( myids )/3 )
  mygroup = c( rep("G0",groupn), rep("G1",groupn), rep("G2",groupn) )
  sampler = sample( 1:length(myids), n, replace=TRUE )
  mydf = data.frame(
    Id = myids[ sampler ],
    DX = mygroup[ sampler ],
    quality = rnorm( n ),
    RandomBasisProjection01 = rnorm( n ),
    RandomBasisProjection02 = rnorm( n ),
    RandomBasisProjection03 = rnorm( n ),
    RandomBasisProjection04 = rnorm( n ),
    RandomBasisProjection05 = rnorm( n ),
    RandomBasisProjection06 = rnorm( n ) )

  # now distribute visits over each subject
  visit = rep( NA, n )
  vizzes = c("V0","V1","V2")
  for ( u in myids ) {
    losel = mydf$Id == u
    loseln = sum( losel )
    loviz = sample( vizzes, loseln, replace=TRUE )
    visit[ losel ] = loviz
  }
  mydf$visit = visit

  # now distribute repeats over each subject
  repeati = rep( NA, n )
  for ( u in myids ) {
    for ( v in vizzes ) {
      losel = mydf$Id == u & mydf$visit == v
      loseln = sum( losel )
      lorep = paste0("Rep",formatC(1:loseln, width = 3, format = "d", flag = "0"))
      repeati[ losel ] = lorep
    }
  }
  mydf$repeater = repeati

  cognition = rep( NA, n )
  ct = 1
  for ( v in vizzes ) {
    for ( k in unique( mygroup ) ) {
      losel = mydf$DX == k & mydf$visit == v
      cognition[ losel ] = rnorm( sum(losel), mean = groupmeansbytime[ct] )
      ct = ct + 1
    }
  }
  mydf$cognition = cognition
  return( mydf )
}


#' Undersample a majority class in a data frame
#'
#' This function undersamples the majority class in a data frame to match the size of the minority class.
#'
#' @param x data frame containing the data to be undersampled
#' @param vname character string specifying the name of the column containing the class labels
#'
#' @return data frame with the majority class undersampled to match the size of the minority class
#'
#' @examples
#' data <- data.frame(class = c(rep("A", 10), rep("B", 100)), 
#'                    feature1 = rnorm(110), 
#'                    feature2 = rnorm(110))
#' undersampled_data <- undersample_majority(data, "class")
#' @export 
undersample_majority <- function(x, vname) {
  # Calculate the proportion of the minority class
  p <- sort(prop.table(table(x[, vname])))[1]
  
  # Split the data into minority and majority classes
  xsmall <- x[x[, vname] == names(p), ]
  xbig <- x[x[, vname] != names(p), ]
  
  # Undersample the majority class to match the size of the minority class
  xbig_undersampled <- xbig[sample(nrow(xbig), size = nrow(xsmall), replace = FALSE), ]
  
  # Combine the minority class with the undersampled majority class
  return(rbind(xsmall, xbig_undersampled))
}

#' Oversample a minority class in a data frame
#'
#' This function oversamples the minority class in a data frame to match the size of the majority class.
#'
#' @param x data frame containing the data to be oversampled
#' @param vname character string specifying the name of the column containing the class labels
#'
#' @return data frame with the minority class oversampled to match the size of the majority class
#'
#' @examples
#' data <- data.frame(class = c(rep("A", 10), rep("B", 100)), 
#'                    feature1 = rnorm(110), 
#'                    feature2 = rnorm(110))
#' oversampled_data <- oversample_minority(data, "class")
#' @export 
oversample_minority <- function(x, vname) {
  # Calculate the proportion of the minority class
  p <- sort(prop.table(table(x[, vname])))[1]
  
  # Split the data into majority and minority classes
  xbig <- x[x[, vname] != names(p), ]
  xsmall <- x[x[, vname] == names(p), ]
  
  # Oversample the minority class to match the size of the majority class
  xsmall_oversampled <- xsmall[sample(nrow(xsmall), size = nrow(xbig), replace = TRUE), ]
  
  # Combine the oversampled minority class with the majority class
  return(rbind(xbig, xsmall_oversampled))
}

#' Plot longitudinal trajectories across subtypes
#'
#' Uses `ggplot2` to visualize differences between subtypes over time.
#'
#' @param mxdfin Input data frame with repeated measurements and a grouped time variable
#' @param idvar variable name for unique subject identifier column
#' @param measurement the numeric variable to plot over time stratified by subtype
#' @param subtype the categorical (possibly ordinal) subtype variable
#' @param vizname the name of the grouped time variable (e.g. years change rounded to nearest quarter year)
#' @param whiskervar character either ci or se
#' @param extra additional naming variable
#' @param xlab additional naming variable
#' @param ylab additional naming variable
#'
#' @return ggplot
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' summ = plotSubtypeChange( mydf, "Id", "cognition", "DX", "visit" )
#' @export
#' @importFrom imbalance mwmote oversample racog rwo
#' @importFrom dCUR CUR
#' @importFrom MASS ginv
#' @importFrom dplyr arrange
#' @importFrom mclust Mclust predict.Mclust densityMclust
#' @importFrom visreg visreg
#' @importFrom dplyr sym bind_rows
#' @importFrom caret createDataPartition 
#' @importFrom pheatmap pheatmap
#' @importFrom gprofiler2 gost gsnpense
#' @importFrom NMF nmf basismap
#' @importFrom stats lm predict qt rnorm var na.omit kmeans
#' @importFrom DDoutlier  LOOP  LOF  INFLO  RDOS  KDEOS  LDF  KNN_AGG  KNN_IN  KNN_SUM  RKOF
#' @importFrom ggplot2 aes ylim guides theme_bw scale_colour_hue geom_errorbar position_dodge element_text geom_line geom_point ggplot guide_legend
#' @importFrom ggplot2 xlab ylab theme rel geom_violin geom_boxplot scale_colour_manual
#' @importFrom plyr ddply rename
#' @importFrom grDevices dev.off pdf
plotSubtypeChange <-function( mxdfin,
                           idvar,
                           measurement,
                           subtype,
                           vizname='timer', # should be roundable to integer
                           whiskervar = c('ci','se'),
                           extra = "",
                           xlab = '',
                           ylab = ''
                          )
    {
      stopifnot( idvar %in% colnames(mxdfin) )
      stopifnot( measurement %in% colnames(mxdfin) )
      stopifnot( subtype %in% colnames(mxdfin) )
      stopifnot( vizname %in% colnames(mxdfin) )

      if ( ! all( whiskervar %in% c("ci","se") ) )
        stop("Must choose ci or se as a string for whiskervar")

      mysd <- function( x,na.rm ) sqrt(var(x,na.rm=na.rm))

      summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=TRUE,
                            conf.interval=.95, .drop=TRUE) {
          # New version of length which can handle NA's: if na.rm==T, don't count them
          length2 <- function (x, na.rm=FALSE) {
              if (na.rm) sum(!is.na(x))
              else       length(x)
          }

          # This does the summary. For each group's data frame, return a vector with
          # N, mean, and sd
          datac <- plyr::ddply(data, groupvars, .drop=.drop,
            .fun = function(xx, col) {
              c(N    = length2(xx[[col]], na.rm=TRUE),
                mean = mean   (xx[[col]], na.rm=TRUE),
                sd   = mysd     (xx[[col]], na.rm=TRUE)
              )
            },
            measurevar
          )
          # Rename the "mean" column
          datac <- plyr::rename(datac, c("mean" = measurevar))

          datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean

          # Confidence interval multiplier for standard error
          # Calculate t-statistic for confidence interval:
          # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
          ciMult <- qt(conf.interval/2 + .5, datac$N-1)
          datac$ci <- datac$se * ciMult

          return(datac)
      }


    lmxdf = mxdfin
    lmxdf$timer = lmxdf[ , vizname ]
# When the population standard deviation is known, the formula for a confidence interval (CI) for a population mean is x̄ ± z* σ/√n, where x̄ is the sample mean, σ is the population standard deviation, n is the sample size, and z* represents the appropriate z*-value from the standard normal distribution for your desired confidence level.

    tgcWithin <- summarySE( lmxdf, measurevar=measurement,
            groupvars=c('timer',subtype), na.rm=TRUE, conf.interval=.95 )

    colnames( tgcWithin )[1:5] = c('timer',"Quant","N","Changer","Changer_norm")
    tgcWithin$ymin = tgcWithin[,"Changer"] - tgcWithin[,whiskervar[1]]
    tgcWithin$ymax = tgcWithin[,"Changer"] + tgcWithin[,whiskervar[1]]
    yran = range( c( tgcWithin$ymin,tgcWithin$ymax ) )
    pd <- ggplot2::position_dodge( 0.05 ) # move them .05 to the left and right
    ww = 0.5
    namer=paste( subtype, ' vs ', measurement , extra )
    return( ggplot( tgcWithin,
      aes(x=timer, y=Changer,
        group = Quant, color = Quant  )) + #, shape = Quant
      geom_errorbar(aes(ymin=ymin, ymax=ymax), colour="black", width=ww, position=pd) +
      geom_line(lwd = 1, show.legend = FALSE ) +
      geom_point(position=pd, size=3, shape=21, fill="white") + # 21 is filled circle
      xlab( xlab ) + ylab( paste(ylab, measurement," +/-", whiskervar[1] )) +
      ylim(yran[1], yran[2]) +
      theme(strip.text = element_text(size = rel(1.1))) +
      theme(legend.position = 'top') +
      guides(fill=guide_legend(title.position="top")) +
      scale_colour_hue(name=subtype, l=40) + # Use darker colors, lightness=40
        theme_bw() +
        theme(text = element_text(size=20),
            axis.text.x = element_text(angle=0, vjust=1),
            legend.direction = "horizontal", legend.position = "top" )
      )

#    return( tgcWithin )
    }



#' Get subjects within a given set of visits
#'
#' @param mxdfin Input data frame with repeated measurements and a grouped time variable
#' @param visitvar variable name for the visit column
#' @param visitselect the desired visits
#'
#' @return data frame
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' mydf = filterByVisit( mydf,  "visit", c("V0","V1"))
#' @export
filterByVisit  <-function(
  mxdfin,
  visitvar,
  visitselect ) {

return( mxdfin[ mxdfin[,visitvar] %in% visitselect, ] )

}




#' Get subjects and timepoints with the best quality
#'
#' This filters data that has both repeats (e.g. multiple images on the same day)
#' and (optionally) longitudinal data.  The quality measure should be a "higher
#' is better" measurement.
#'
#' @param mxdfin Input data frame with repeated measurements and a grouped time variable
#' @param idvar variable name for unique subject identifier column
#' @param visitvar variable name for the visit or date column
#' @param qualityvar variable name for the quality column; higher values should map to
#' higher quality data.
#'
#' @return data frame
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' mydfhq = highestQualityRepeat( mydf, "Id", "visit", "quality")
#' @export
highestQualityRepeat  <-function(
  mxdfin,
  idvar,
  visitvar,
  qualityvar ) {

  if ( ! ( visitvar %in% names( mxdfin ) ) ) stop("visitvar not in dataframe")
  if ( ! ( idvar %in% names( mxdfin ) ) ) stop("idvar not in dataframe")
  if ( ! ( qualityvar %in% names( mxdfin ) ) ) stop("qualityvar not in dataframe")
  vizzes = unique( mxdfin[,visitvar] )
  uids = unique( mxdfin[,idvar] )
  useit = rep( FALSE, nrow( mxdfin ) )
  for ( u in uids ) {
    losel = mxdfin[,idvar] == u
    vizzesloc = unique( mxdfin[ losel, visitvar ] )
    for ( v in vizzesloc ) {
      losel = mxdfin[,idvar] == u & mxdfin[,visitvar] == v
      mysnr = mxdfin[losel,qualityvar]
      myw = which( losel )
      if ( length( myw ) > 1 ) {
        if ( any( !is.na(mysnr) )  )
          useit[ myw[ which.max(mysnr) ] ] = TRUE
        } else useit[ myw ] = TRUE
      }
    }
  return( mxdfin[ useit, ] )
}


#' Reject subjects and timepoints with lowest quality
#'
#' This filters data that has both repeats (e.g. multiple images on the same day)
#' and (optionally) longitudinal data.  The quality measure should be a "higher
#' is better" measurement.  This function can be called recursively until it
#' converges.  It is less aggressive than \code{highestQualityRepeat}.
#'
#' @param mxdfin Input data frame with repeated measurements and a grouped time variable
#' @param idvar variable name for unique subject identifier column
#' @param visitvar variable name for the visit or date column
#' @param qualityvar variable name for the quality column; higher values should map to
#' higher quality data.
#' @param verbose boolean
#'
#' @return data frame
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' mydfhq = rejectLowestQualityRepeat( mydf, "Id", "visit", "quality")
#' @export
rejectLowestQualityRepeat <-function(
  mxdfin,
  idvar,
  visitvar,
  qualityvar,
  verbose = FALSE ) {

  if ( ! ( visitvar %in% names( mxdfin ) ) ) stop("visitvar not in dataframe")
  if ( ! ( idvar %in% names( mxdfin ) ) ) stop("idvar not in dataframe")
  if ( ! ( qualityvar %in% names( mxdfin ) ) ) stop("qualityvar not in dataframe")
  vizzes = sort(unique( mxdfin[,visitvar] ))
  uids = sort(unique( mxdfin[,idvar] ))
  useit = rep( TRUE, nrow( mxdfin ) )
  for ( u in uids ) {
    losel = mxdfin[,idvar] == u
    vizzesloc = unique( mxdfin[ losel, visitvar ] )
    for ( v in vizzesloc ) {
      losel = mxdfin[,idvar] == u & mxdfin[,visitvar] == v
      mysnr = mxdfin[losel,qualityvar]
      myw = which( losel )
      if ( length( myw ) > 1 )
        if ( any( !is.na(mysnr) )  )
          useit[ myw[ which.min(mysnr) ] ] = FALSE
      }
    }
  if ( verbose ) print( table( useit ) )
  return( mxdfin[ useit, ] )
}



#' Average repeat data
#'
#' @param mxdfin Input data frame with repeated measurements and a grouped time variable
#' @param idvar variable name for unique subject identifier column
#' @param visitvar variable name for the visit or date column
#'
#' @return data frame
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' mydfhq = averageRepeats( mydf, "Id", "visit")
#' @export
averageRepeats  <-function(
  mxdfin,
  idvar,
  visitvar ) {

  if ( ! ( visitvar %in% names( mxdfin ) ) ) stop("visitvar not in dataframe")
  if ( ! ( idvar %in% names( mxdfin ) ) ) stop("idvar not in dataframe")
  vizzes = unique( mxdfin[,visitvar] )
  uids = unique( mxdfin[,idvar] )
  useit = rep( FALSE, nrow( mxdfin ) )
  num_cols <- unlist(lapply(mxdfin, is.numeric))
  for ( u in uids ) {
    losel = mxdfin[,idvar] == u
    vizzesloc = unique( mxdfin[ losel, visitvar ] )
    for ( v in vizzesloc ) {
      losel = mxdfin[,idvar] == u & mxdfin[,visitvar] == v
      myw = which( losel )
      if ( length( myw ) > 1 ) {
          useit[ myw[ 1 ] ] = TRUE
          mxdfin[ myw[ 1 ] , num_cols ] = colMeans( mxdfin[ myw, num_cols ], na.rm=TRUE )
        } else useit[ myw ] = TRUE
      }
    }
  return( mxdfin[ useit, ] )
}



#' Outlierness scoring
#'
#' Produce several complementary measurements of outlierness
#'
#' @param mxdfin Input data frame with repeated measurements and a grouped time variable
#' @param measureColumns vector string of column names to use in outlier detection
#' @param calck optional integer for knn
#' @param outlierfunctions vector of strings naming outlier functions to report
#' @param verbose boolean
#' @return data frame
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' rbfnames = names(mydf)[grep("Random",names(mydf))]
#' mydf[8,rbfnames] = mydf[8,rbfnames] * 4.0
#' mydfol = outlierness( mydf, rbfnames )
#' olnames = names(mydf)[grep("OL_",names(mydf))]
#' mydf[6,olnames]
#' mydf[8,olnames]
#' @export
#' @importFrom stats quantile
#' @importFrom utils tail
outlierness <- function( mxdfin, measureColumns, calck,
  outlierfunctions = c("LOOP", "LOF",  "INFLO", "RDOS", "KDEOS",
    "LDF", "KNN_AGG", "KNN_IN", "KNN_SUM", "RKOF" ),
  verbose=FALSE ) {
  olnames = c("OL_LOOP", "OL_LOF", "OL_NOF", "OL_INFLO", "OL_RDOS", "OL_KDEOS",
    "OL_LDF", "OL_KNN_AGG", "OL_KNN_IN", "OL_KNN_SUM", "OL_RKOF" )
  oldata = scale( mxdfin[,measureColumns], TRUE, TRUE )
  if ( missing( calck ) )
    calck = DDoutlier::NAN( oldata )$r # this + LOOP is good
  for ( myolf in outlierfunctions ) {
    if ( myolf == "LOOP" ) {
      if ( verbose ) cat( "calc: LOOP ... ")
      mxdfin$OL_LOOP = DDoutlier::LOOP(  oldata, k=calck )
    } else if ( myolf == "LOF"  ) {
      if ( verbose ) cat( "calc: LOF ... ")
      mxdfin$OL_LOF = DDoutlier::LOF(  oldata, k=calck )
    } else if ( myolf == "INFLO"  ) {
      if ( verbose ) cat( "calc: INFLO ... ")
      mxdfin$OL_INFLO = ( DDoutlier::INFLO( oldata, k=calck ) )
    } else if ( myolf == "NOF" ) {
      if ( verbose ) cat( "calc: NOF ... ")
      mxdfin$OL_NOF = DDoutlier::NOF( oldata )$NOF
    } else if ( myolf == "KDEOS" ) {
      if ( verbose ) cat( "calc: KDEOS ... ")
      mxdfin$OL_KDEOS = DDoutlier::KDEOS( oldata, k_min = round( calck * 0.5 ), k_max = calck + 1 )
    } else if ( myolf == "LDF" ) {
      if ( verbose ) cat( "calc: LDF ... ")
      mxdfin$OL_LDF = DDoutlier::LDF( oldata, k = calck  )$LDF
    } else if ( myolf == "KNN_AGG" ) {
      if ( verbose ) cat( "calc: KNN_AGG ... ")
      mxdfin$OL_KNN_AGG = DDoutlier::KNN_AGG( oldata, k_min = round( calck * 0.5 ), k_max = calck + 1  )
    } else if ( myolf == "KNN_IN" ) {
      if ( verbose ) cat( "calc: KNN_IN ... ")
      mxdfin$OL_KNN_IN = -1.0 * DDoutlier::KNN_IN( oldata, k = calck )
    } else if ( myolf == "KNN_SUM" ) {
      if ( verbose ) cat( "calc: KNN_SUM ... ")
      mxdfin$OL_KNN_SUM = DDoutlier::KNN_SUM( oldata, k = calck  )
    } else if ( myolf == "RKOF" ) {
      if ( verbose ) cat( "calc: RKOF ... ")
      mxdfin$OL_RKOF = DDoutlier::RKOF( oldata, k = calck  )
    }
  }
  return( mxdfin )
}




#' Carefully replace a column name with a new one
#'
#' This will replace a column name with a new column name - it will not make
#' any changes to the underlying data.
#'
#' @param dataIn Input data frame with the old name within
#' @param oldName string column name to replace
#' @param newName the replacement name
#' @return data frame
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' rbfnames = names(mydf)[grep("Random",names(mydf))]
#' mydf = replaceName( mydf, rbfnames[1], 'mileyCyrus' )
#' @export
replaceName <- function( dataIn, oldName, newName ) {
  if ( oldName %in% names( dataIn ) )
    names(dataIn)[ names(dataIn) == oldName ] = newName
  return( dataIn )
}


#' Find test-retest data within a dataframe
#'
#' We assume there is a quality or related measure that will let us compute
#' distances between different time points and that these distances relate to
#' how similar two images are.  Note that the measurement may not relate to
#' quality at all but should map images into a similar metric space based on
#' appearance or some other objective quality that leads to the same value
#' when images are the same and changes continuously as images differ.
#'
#' @param dataIn Input data frame with the old name within
#' @param qualitycolumns the name(s) of quality-related column measurements
#' @param subjectID the unique subject id column name
#' @param visitID the column name that gives the visit / date; typically we look
#' for data the exists on the same visit.
#' @param whichVisit optional visit to use for the analysis
#' @param measureToRepeat the measurement to assemble for computing trt stats (e.g. ICC)
#' @param uniqueID optional column to contatenate to trt dataframe
#' @param covariates optional additional column name(s) to add to dataframe
#' @param nozeroes optional boolean - do not allow zero distance time
#' @return data frame with test-retest friendly organization ... trt0 and trt1 show the row indices of the test retest data
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' rbfnames = names(mydf)[grep("Random",names(mydf))]
#' mytrt = assembleTestRetest( mydf, rbfnames[1], 'Id', 'visit' )
#' @export
assembleTestRetest <- function(
  dataIn,
  qualitycolumns,
  subjectID,
  visitID,
  whichVisit,
  measureToRepeat,
  uniqueID, 
  covariates,
  nozeroes=FALSE ) {

  usubs = sort( unique( dataIn[,subjectID] ) )
  trtdf = data.frame( )
  for ( u in usubs ) {
    usel = dataIn[,subjectID] == u
    if ( ! missing( whichVisit ) ) {
      usel = dataIn[,subjectID] == u & dataIn[,visitID] == whichVisit
    }
    usel[ is.na( usel ) ] = FALSE
    if (sum(usel) > 1 ) {
      rowinds = which( usel )
      mydist = as.matrix( dist( dataIn[usel,qualitycolumns] ) )
      diag(mydist) = Inf
      if ( nozeroes ) mydist[ mydist == 0 ] = Inf
      mymin = min(mydist,na.rm=TRUE)
      bestind = which( mydist == mymin, arr.ind = T)[1,]
      n = nrow(trtdf) + 1
      trtdf[ n , subjectID ] = u
      if ( ! missing( whichVisit ) )
        trtdf[ n , visitID ] = whichVisit
      trtdf[ n , "distance" ] = mymin
      trtdf[ n , "trt0" ] = rowinds[ bestind[1] ]
      trtdf[ n , "trt1" ] = rowinds[ bestind[2] ]
      if ( ! missing( measureToRepeat )) {
        trtdf[ n , paste0(measureToRepeat,0) ] = dataIn[ trtdf[ n , "trt0" ], measureToRepeat ]
        trtdf[ n , paste0(measureToRepeat,1) ] = dataIn[ trtdf[ n , "trt1" ], measureToRepeat ]
      }
      if ( ! missing( uniqueID )) {
        trtdf[ n , paste0(uniqueID,0) ] = dataIn[ trtdf[ n , "trt0" ], uniqueID ]
        trtdf[ n , paste0(uniqueID,1) ] = dataIn[ trtdf[ n , "trt1" ], uniqueID ]
      }
      if ( ! missing( covariates ) ) {
        for (covariate in covariates) {
          trtdf[ n , paste0(covariate,"_0") ] = dataIn[ trtdf[ n , "trt0" ], covariate ]
          trtdf[ n , paste0(covariate,"_1") ] = dataIn[ trtdf[ n , "trt1" ], covariate ]
        }
      } 
    }
  }
  return( trtdf )
}


#' Uses outlierness scoring to filter a data frame
#'
#' This will produce an inclusion function based on scores genetrated by the
#' \code{outlinerness} function within this package.  We assume the data frame
#' has values of the type \code{RandBasisProj*} or \code{RandomBasisProj*} within its column names.
#'
#' @param dataIn Input data frame
#' @param bestolvar the outlier variable to prioritize
#' @param quantileThreshold the quantile to use for thresholing outlierness values
#' @param extraFiltering will perform brain-based application-specific filtering
#' @param calck optional parameter to pass to outlierness
#' @return data frame
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' mydfol = filterForGoodData( mydf, calck = 8 )
#' @export
#' @importFrom stats quantile
#' @importFrom utils tail
filterForGoodData <- function( dataIn,
    bestolvar = "OL_RKOF",
    quantileThreshold=0.99,
    extraFiltering = FALSE,
    calck = 256 ) {
  rbj = "RandBasisProj"
  if ( length( grep(rbj,names(dataIn)) ) == 0 ) {
    rbj = "RandomBasisProj"
    if ( length( grep(rbj,names(dataIn)) ) == 0 ) {
      stop("No RandBasisProj or RandomBasisProj column names. Cannot proceed.")
    }
  }
  epshvbrain_adjusted = 1.0 - quantileThreshold
  goodsel = rep( TRUE, nrow( dataIn ) )
  if ( ( "dkt_parc_wmSNR_dkt_parc_wmSNR" %in% names( dataIn ) ) & extraFiltering ) {
    qval = quantile( dataIn$dkt_parc_wmSNR_dkt_parc_wmSNR, epshvbrain_adjusted, na.rm=T  )
    goodsel = goodsel & dataIn$dkt_parc_wmSNR_dkt_parc_wmSNR > qval
  }
  if ( ( "GM_vol_tissues" %in% names( dataIn ) ) & extraFiltering ) {
    qval = quantile( dataIn$GM_vol_tissues, c(epshvbrain_adjusted, 1.0 - epshvbrain_adjusted), na.rm=T  )
    goodsel = goodsel & dataIn$GM_vol_tissues > qval[1]
  }

  rbvars = sort( names(dataIn)[grep(rbj,names(dataIn))] )
  goodsel = goodsel[  !is.na( dataIn[,rbvars[1]]) ]
  dataIn = dataIn[ !is.na( dataIn[,rbvars[1]]),  ]
  if ( length( grep("Pos",rbvars) ) > 0 )
    rbvars = rbvars[ -grep("Pos",rbvars)]
  dataIn = outlierness( dataIn, rbvars, calck=calck, verbose = TRUE)
  olvar = names(dataIn)[grep("OL_",names(dataIn)) ]
  goodsel = goodsel &
    dataIn[,bestolvar] < quantile( dataIn[,bestolvar], quantileThreshold, na.rm=T) &
    !is.na(   dataIn[,bestolvar])
  olvarinds = 1:length( olvar )
  olvarinds = olvarinds[ ! ( olvarinds %in% which( olvar == bestolvar ) ) ]
  for ( k in olvarinds ) {
    if ( ! is.na( dataIn[1,olvar[k]] ) ) {
      nextsel = dataIn[,olvar[k]] < quantile( dataIn[,olvar[k]], quantileThreshold, na.rm=T )
      goodsel = goodsel & nextsel
      }
    }
  goodsel[ is.na( goodsel )] = FALSE
  return( dataIn[ goodsel ,  ] )
}



#' fill baseline column
#'
#' build a column of data that maps baseline values to every row. baseline values
#' starting points or reference points for each subject, e.g. a value of a measurement
#' at time zero.  the method will produce a mean value if multiple entries match
#' the subjectID and visitID conditions.
#'
#' @param mxdfin Input data frame with repeated measurements and a grouped time variable
#' @param columnName string defining valid columns in the data frame.
#' @param subjectID the unique subject id column name
#' @param visitID names of the column that defines the variable defining
#' baseline-ness. e.g. \code{isBaseline}
#' @param baselineVisitValue the value defining baseline e.g. \code{TRUE}
#' @param baselineExt string appended to column name defining the baseline variable
#' @param deltaExt string appended to column name defining the change variable
#' @param fast boolean; will only return subjects with baseline values; if there 
#' are several baseline entries, these will be averaged. only works with numeric data.
#' @param verbose boolean
#' @return data frame with new columns
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' mydf = fillBaselineColumn( mydf, "RandomBasisProjection01", "Id", "visit", "V0" )
#' @export
fillBaselineColumn <- function(
   mxdfin,
   columnName,
   subjectID,
   visitID,
   baselineVisitValue,
   baselineExt = "_BL",
   deltaExt = "_delta",
   fast = TRUE,
   verbose=TRUE
) {

  myMode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
    }
  mostfreq <- function( x, na.rm=TRUE ) {
    if ( is.numeric( x ) ) return( mean(x,na.rm=na.rm ) )
    return( myMode( x ) )
  }
  if ( ! ( subjectID %in% names( mxdfin ) ) )
    stop("subjectID not in data frame's columns")
  if ( ! all( columnName %in% names( mxdfin ) ) )
    stop("columnName not in data frame's columns")
  if ( ! ( visitID %in% names( mxdfin ) ) )
    stop("visitID not in data frame's columns")
  if ( ! ( baselineVisitValue %in% mxdfin[,visitID] ) )
    stop("baselineVisitValue not in data frame's visitID")
  newcolname = paste0( columnName, baselineExt )
  newcolnamed = paste0( columnName, deltaExt )
  if ( fast ) { # use columnful merge (was use_data.table)
    # get the sid freq
    mxdfin[,subjectID]=as.character(mxdfin[,subjectID])
    mxdfin = mxdfin[ order( mxdfin[,subjectID] ), ]
    sidfreq = table( mxdfin[,subjectID])
    bldf = mxdfin[ fs(mxdfin[,visitID] == baselineVisitValue),  c(subjectID,columnName) ]
    sidtblbl = table( bldf[,subjectID]) 
    maxbln = max( sidtblbl )
    if ( maxbln > 1 ) {
      onlyones = names( sidtblbl )[ sidtblbl == 1 ]
      morethanones = names( sidtblbl )[ sidtblbl > 1 ]
      bldf1 = bldf[ bldf[,subjectID] %in% onlyones, ]
      sel2 = bldf[,subjectID] %in% morethanones
      if ( verbose )
        print( paste( "aggregate ", columnName, " in ", sum(sel2), "subjects with > 1 row ... these subjects have as many as", maxbln, "entries" ) )
      multisubs=bldf[sel2,subjectID]
      bldf2 = aggregate( bldf[sel2,c(subjectID,columnName)], list(multisubs), mostfreq, na.rm=T)
      bldf2[,subjectID]=unique( multisubs )
      bldf = base::rbind( bldf1[,c(subjectID,columnName)], bldf2[,c(subjectID,columnName)] )
      bldf = bldf[ order( bldf[,subjectID] ), ]
      if ( verbose )
        print("aggregation done")
      }
    inmcols = colnames( bldf ) %in% columnName
    colnames( bldf )[ inmcols ] = newcolname
    mxdfin = mxdfin[ mxdfin[,subjectID] %in% bldf[,subjectID], ]
    sidfreq = sidfreq[ names(sidfreq) %in% bldf[,subjectID] ]
    sidfreq = sidfreq[ as.character(bldf[,subjectID]) ]
    rownames(bldf)=bldf[,subjectID]
    if ( verbose ) {
      print( "begin repeat")
      print( as.integer(sidfreq) )
    }
    bldf = bldf[rep.int( bldf[,subjectID] , as.integer(sidfreq)), ]
    if ( verbose ) {
      print("end repeat/begin fill")
      print(dim(bldf))
      print(dim(mxdfin))
    }
    if ( verbose ) {
      print("selection done")
      print(dim(mxdfin))
      print(dim(bldf))
    }
    if ( identical( as.character(mxdfin[,subjectID]), as.character(bldf[,subjectID]) ) ) {
      mxdfin[,newcolname]=bldf[,newcolname]
    } else {
      print("Subject IDs are not identical")
      print(head(mxdfin[,subjectID]))
      print(head(bldf[,subjectID]))
      print(table( mxdfin[,subjectID]!=bldf[,subjectID]  ))
      stop("Subject IDs are not identical")
    }
    if ( verbose )
      print("end fill")
#    filldf = dplyr::bind_rows( mxdfin, bldf, .id=subjectID )
#    filldf = data.table(mxdfin)[bldf, on = subjectID, allow.cartesian=FALSE] # data.table
    if ( ! is.na( deltaExt ) ) {
      mxdfin[,newcolnamed] = mxdfin[,columnName] - mxdfin[,newcolname]
    }
    return( list(mxdfin, newcolname, newcolnamed ) )
  }
  mxdfin[,newcolname]=NA
  mxdfin[,newcolnamed]=NA
  visitidisnumeric = is.numeric(mxdfin[,visitID])
  usubs = unique( mxdfin[,subjectID] )
  nsubs = length( usubs )
  ct=0
  for ( u in usubs ) {
    if ( ct %% 20 == 0 ) cat( paste0(round(ct/nsubs*100),"%.") )
    ct=ct+1
    losel = fs( mxdfin[,subjectID] == u )
    lomxdfin = mxdfin[ losel ,  ]
    selbase = fs( lomxdfin[,visitID] == baselineVisitValue )
    if ( sum(selbase) == 0 & visitidisnumeric ) { # take next best value
      minval = min( lomxdfin[,visitID],na.rm=T )
      selbase = fs(lomxdfin[,visitID] == minval)
      }
    selbase[ is.na(selbase) ] = FALSE
    for ( jj in 1:length( columnName ) ) {
      columnNameLoc = columnName[jj]
      newcolnameLoc = newcolname[jj]
      isFactor = class( lomxdfin[,columnNameLoc] ) != "numeric"
      baseval = NA
      if ( sum( selbase ) > 0  & !isFactor ) {
        baseval = mean( lomxdfin[ selbase, columnNameLoc ],  na.rm=T )
      } else if ( sum( selbase ) > 0  & isFactor ) {
        baseval = median( lomxdfin[ selbase, columnNameLoc ],  na.rm=T )
      }
      mxdfin[ losel , newcolnameLoc ] = baseval
    }
  }
  if ( ! is.na( deltaExt ) )
    mxdfin[,newcolnamed] = mxdfin[,columnName] - mxdfin[,newcolname]
  return( list(mxdfin, newcolname, newcolnamed ) )
}


#' Covariate adjustment
#'
#' Adjust a training vector value by nuisance variables eg field strength etc.
#' One may want to use a specific sub-group for this, e.g. controls.
#'
#' @param mxdfin Input data frame with repeated measurements and a grouped time variable
#' @param adjustmentFormula string defining a valid formula to be used in \code{lm}.
#' @param groupVariable names of the column that defines the group to use for training.
#' @param group string defining a subgroup on which to train
#' @param allowMissing boolean
#' @return data frame with adjusted measurement variable as defined in formula
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' myform = "cognition ~ RandomBasisProjection01 "
#' mydf = adjustByCovariates(mydf,myform,"DX","G0")
#' @export
adjustByCovariates  <- function(
  mxdfin,
  adjustmentFormula,
  groupVariable,
  group,
  allowMissing=FALSE
) {
  outcomevar = gsub( " ", "", unlist(strsplit( adjustmentFormula, "~" ))[[1]] )
  if ( ! missing( group ) & ! missing( groupVariable ) ) {
    if ( ! (groupVariable %in% names( mxdfin ) ) ) stop("group name is wrong")
    gsel = mxdfin[,groupVariable] %in% group
    if ( sum(gsel) < 5 ) stop("too few subjects in subgroup for training")
    subdf = mxdfin[ gsel, ]
  } else subdf = mxdfin
  for ( xx in all.vars( as.formula( adjustmentFormula ) ) ) {
#    print( xx )
 #   print( table( is.na( mxdfin[,x] ) ) )
  }
  baseform = paste( outcomevar, " ~ 1" )
  imodel = lm( baseform, data=subdf ) # just the intercept
  ctlmodel = lm( adjustmentFormula, data=subdf )
#  print( table(subdf$istrain, subdf$studyName ) )
#  print( table(mxdfin$istrain, mxdfin$studyName ) )
  predintercept = predict( imodel, newdata = mxdfin  )
  predvol = predict( ctlmodel, newdata = mxdfin ) - predintercept
  adjustedoutcome = paste0( outcomevar, "_adjusted" )
  if ( ! allowMissing ) {
    mxdfin[ , adjustedoutcome ] = mxdfin[,outcomevar] - predvol 
  } else {
    wtoadjust = !is.na(predvol)
    mxdfin[ , adjustedoutcome ] = mxdfin[,outcomevar]
    mxdfin[ wtoadjust, adjustedoutcome ] = mxdfin[wtoadjust,outcomevar] - predvol[wtoadjust] 
  }
  return( mxdfin )
}


#' Covariate adjustment for many variables based on a single variable
#'
#' Adjust a training vector value by nuisance variables eg field strength etc.
#' One may want to use a specific sub-group for this, e.g. controls.
#'
#' @param mxdfin Input data frame with repeated measurements and a grouped time variable
#' @param adjustmentFormula string defining a valid formula to be used in \code{lm}.
#' @param columnstoadjust a vector of strings
#' @param groupVariable names of the column that defines the group to use for training.
#' @param group string defining a subgroup on which to train
#' @return data frame with adjusted measurement variable as defined in formula
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' myform = "cognition ~ RandomBasisProjection01 "
#' mydf = adjustByCovariates(mydf,myform,"DX","G0")
#' @export
adjustByCovariatesUni  <- function(
  mxdfin,
  adjustmentFormula,
  columnstoadjust,
  groupVariable,
  group
) {
  ##############################################
  cstrained <- function( x, istrain ) {
    xout = rep(NA,length(x))
    scaledTrainData = scale(x[istrain])
    xout[istrain] = scaledTrainData
    xout[!istrain] = scale(x[!istrain], 
        center=attr(scaledTrainData, "scaled:center"), 
        scale=attr(scaledTrainData, "scaled:scale"))
    return( xout )
  }
  outcomevar = gsub( " ", "", unlist(strsplit( adjustmentFormula, "~" ))[[1]] )
  if ( ! (groupVariable %in% names( mxdfin ) ) ) stop("group name is wrong")
  gsel = mxdfin[,groupVariable] %in% group
  if ( sum(gsel) < 5 ) stop("too few subjects in subgroup for training")
  subdf = mxdfin
  subdf$adjustByCovariatesUniTempVar = cstrained( 
    subdf[,outcomevar], gsel )
  adjustmentFormula2 = gsub(outcomevar, "adjustByCovariatesUniTempVar", adjustmentFormula )
  ctlmodel = lm( adjustmentFormula2, data=subdf[gsel,] )
  predvol = predict( ctlmodel, newdata = subdf )
  for ( zz in columnstoadjust ) {
    adjustedoutcome = paste0( zz, "_adjusted" )
    myscld = cstrained( mxdfin[,zz], gsel )
    mxdfin[ , adjustedoutcome ] = myscld - predvol
    }
  return( mxdfin )
}


#' Train subtype for univariate data
#'
#' This is the training module for subtype definition based on a vector.
#' Produces a data frame denoting the measurement variable, its cutpoints and
#' the names of the subtypes.
#' One may want to use a specific sub-group for this, e.g. patients.
#'
#' @param mxdfin Input data frame
#' @param measureColumn string defining a valid column to use for subtyping.  E.g.
#' the output of the covariate adjustment done by \code{adjustByCovariates}.
#' @param subtypes string vector naming the subtypes
#' @param quantiles numeric vector defining the quantiles that will yield subgroups
#' @param subtypename string naming the subtype variable
#' @param groupVariable names of the column that defines the group to use for training.
#' @param group string defining a subgroup on which to train
#' @return data frame defining the subtypes and cutpoints
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' qdf = trainSubtypeUni( mydf, "cognition", c("C0","C1","C2"), c(0.33,0.66) )
#' @export
trainSubtypeUni  <- function(
  mxdfin,
  measureColumn,
  subtypes,
  quantiles = c(0.25,0.75),
  subtypename = "subtype",
  groupVariable,
  group
) {

  if ( length( subtypes ) != ( length( quantiles ) + 1 ) )
    stop( "length( subtypes ) != ( length( quantiles ) + 1 )" )
  if ( ! missing( group ) & ! missing( groupVariable ) ) {
    if ( ! (groupVariable %in% names( mxdfin ) ) ) stop("group name is wrong")
    gsel = mxdfin[,groupVariable] %in% group
    if ( sum(gsel) < 5 ) stop("too few subjects in subgroup for training")
    subdf = mxdfin[ gsel, ]
  } else {
    subdf = mxdfin
    group=NA
  }

  nrow = length( quantiles ) + 1
  stdf = data.frame(
    subtypes = subtypes,
    measurement = rep( measureColumn, nrow ),
    quantiles = c( 0, quantiles ),
    quantileValues = rep( NA, nrow ),
    group = rep( group, nrow ) )
  names( stdf )[1] = subtypename
  stdf$quantileValues = c( -Inf, quantile( subdf[,measureColumn],quantiles, na.rm=TRUE ) )
  return( stdf )
}

#' Scale Numeric Variables in a Data Frame
#'
#' This function scales the numeric variables in a data frame that are 
#' specified in a given equation. Non-numeric variables mentioned in the 
#' equation are ignored. Scaling is performed by centering and dividing by the 
#' standard deviation.
#'
#' @param mydf A data frame containing the variables to be scaled.
#' @param myeq An equation in the form of a string or an object of class 
#'   \code{formula} specifying the variables to be scaled. Only the variables 
#'   on the right-hand side of the equation are considered.
#' @param variables_to_exclude vector of strings not to scale
#'
#' @return A data frame with the specified numeric variables scaled. The 
#'   function modifies the input data frame in place and returns it.
#'
#' @examples
#' data(iris)
#' scaled_iris <- scale_variables_in_equation(iris, myeq = "Sepal.Length ~ Sepal.Width + Petal.Length")
#' head(scaled_iris)
#'
#' @export
scale_variables_in_equation <- function( mydf, myeq, variables_to_exclude ) {
  myterms = all.vars(as.formula(myeq))[-1]
  if ( ! missing( variables_to_exclude ) )
    myterms = myterms[ ! ( myterms %in% variables_to_exclude )]
  myterms = intersect(myterms, colnames(mydf))
  for ( x in myterms ) {
    if ( is.numeric( mydf[,x] )) {
      if ( var( mydf[,x], na.rm=T ) > 0 )
        mydf[,x] = c(scale(mydf[,x]))
    }
  }
  return(mydf)
}



#' Predict subtype for univariate data
#'
#' This is the inference module for subtype definition based on a vector.
#' After training, we can predict subtype very easily in new data based on
#' the data frame produced by training \code{trainSubtypeUni}.  If one passes
#' the visitName to the function then we will define subtype from the baseline
#' value alone.
#'
#' @param mxdfin Input data frame
#' @param subtypeDataFrame data frame defining the subtypes and cutpoints
#' @param idvar variable name for unique subject identifier column
#' @param visitName the column name defining the visit variables
#' @param baselineVisit the string naming the baseline visit
#' @param rename boolean will rename levels to user-provided names
#' @return data frame with attached subtypess
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' rbfnames = names(mydf)[grep("Random",names(mydf))]
#' mydf = outlierness( mydf, rbfnames )
#' mydf = highestQualityRepeat( mydf, "Id", "visit", "OL_KNN_SUM")
#' qdf = trainSubtypeUni( mydf, "cognition", c("C0","C1","C2"), c(0.33,0.66) )
#' pdf = predictSubtypeUni( mydf, qdf, "Id" )
#' @export
#' @importFrom Hmisc cut2
predictSubtypeUni  <- function(
  mxdfin,
  subtypeDataFrame,
  idvar,
  visitName,
  baselineVisit,
  rename = TRUE ) {

  msr = as.character( subtypeDataFrame[, "measurement"][1] )
  thesubtypes = subtypeDataFrame[,1]
  mxdfin[,names(subtypeDataFrame)[1]] = NA
  quantsV = as.numeric( subtypeDataFrame[,"quantileValues"][-1] )
  quantsP = as.numeric( subtypeDataFrame[,"quantiles"][-1] )
  # by default, just run across each row
  mxdfin[,names(subtypeDataFrame)[1]] = Hmisc::cut2( mxdfin[,msr], cuts = quantsV )
  theselevs = sort( na.omit( unique(  mxdfin[,names(subtypeDataFrame)[1]] ) ) )
  if ( rename ) {
    mxdfin[,names(subtypeDataFrame)[1]] = as.character( mxdfin[,names(subtypeDataFrame)[1]] )
    for ( k in 1:length( theselevs ) ) {
      losel = mxdfin[,names(subtypeDataFrame)[1]] == theselevs[k]
      mxdfin[,names(subtypeDataFrame)[1]][ losel ] = thesubtypes[k]
    }
    mxdfin[,names(subtypeDataFrame)[1]] <- factor(mxdfin[,names(subtypeDataFrame)[1]], levels=thesubtypes )
  } else {
    mxdfin[,names(subtypeDataFrame)[1]] <- factor(mxdfin[,names(subtypeDataFrame)[1]], levels=theselevs )
  }
  if ( missing( visitName ) | missing( baselineVisit ) )
    return( mxdfin )
  if ( ! ( baselineVisit %in% unique( mxdfin[,visitName] ) ) )
    stop( "! ( baselineVisit %in% unique( mxdfin[,visitName] ) )" )
  uids = unique( mxdfin[,idvar] )
  for ( u in uids ) {
    usel = mxdfin[,idvar] == u
    losel0 = mxdfin[,idvar] == u & mxdfin[,visitName] == baselineVisit & !is.na(mxdfin[,msr])
    losel0[ is.na( losel0 ) ] = FALSE
    if ( sum( losel0 ) > 0 ) {
      mytbl = table( mxdfin[losel0,names(subtypeDataFrame)[1]] )
      mostcommon = names( mytbl )[which.max(mytbl)]
      mxdfin[usel,names(subtypeDataFrame)[1]] = mostcommon
    }
  }
  return( mxdfin )

}





#' Train subtype for multivariate data
#'
#' This is the training module for subtype definition based on a matrix.
#' Currently only supports clustering based on gaussian mixtures
#' via the ClusterR package.
#' One may want to use a specific sub-group for this, e.g. patients.
#'
#' @param mxdfin Input data frame
#' @param measureColumns vector defining the data columns to be used for clustering.
#' Note that these methods may be sensitive to scaling so the user may want to
#' scale columns accordingly.
#' @param method string GMM or kmeans or medoid
#' @param desiredk number of subtypes
#' @param maxk maximum number of subtypes
#' @param groupVariable names of the column that defines the group to use for training.
#' @param group string defining a subgroup on which to train
#' @param frobNormThresh fractional value less than 1 indicating the amount of change
#' in the reconstruction error (measured by frobenius norm) from the previous
#' iteration 1 - F_cur / F_prev that will determine the optimal number of clusters.
#' For GMM clustering.
#' @param trainTestRatio Training testing split for finding optimal number
#' of clusters. For GMM clustering.  If zero, then will not split data. Otherwise,
#' will compute reconstruction error in test data only.
#' @param distance_metric see medoid methods in ClusterR
#' @param flexweights optional weights
#' @param flexgroup optional group
#' @param groupFun optional function name to use in group-guided clustering e.g. minSumClusters
#' @return the clustering object
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' rbfnames = names(mydf)[grep("Random",names(mydf))]
#' gmmcl = trainSubtypeClusterMulti( mydf, rbfnames, maxk=4 )
#' @export
#' @importFrom mlr3cluster as_task_clust
#' @importFrom mlr3 mlr_learners
#' @importFrom ClusterR Cluster_Medoids predict_GMM Clara_Medoids GMM Optimal_Clusters_GMM KMeans_rcpp Optimal_Clusters_KMeans Optimal_Clusters_Medoids predict_Medoids
trainSubtypeClusterMulti  <- function(
  mxdfin,
  measureColumns,
  method = 'kmeans',
  desiredk,
  maxk,
  groupVariable,
  group,
  frobNormThresh = 0.01,
  trainTestRatio = 0,
  distance_metric=NULL,
  flexweights=NULL,
  flexgroup=NULL,
  groupFun = NULL
) {

  clustermeth = c("clara","fanny","pam")
  ktypes = c( clustermeth, "kmeans", 'kmeansflex', "GMM", "mclust", "pamCluster", 
    "kmeansflex","kmedians",  "angle",  "ejaccard", "flexcorr","flexcanb","flexmink", "cckmeans", "VarSelCluster",
    "hardcl","neuralgas", "hierarchicalCluster", "jaccard", mlr_learners$keys("clust") )

  if ( ! (method %in% ktypes ) ) {
    allmeth = paste( ktypes, collapse=' | ')
    message(paste("your chosen method:",method, "is not available."))
    message(paste( "try one of:" , allmeth ) )
    return( ktypes )
  }
  ismlr3=FALSE
  if ( length(grep( "clust[.]", method )) > 0 ) ismlr3=TRUE

  .env <- environment() ## identify the environment of cv.step
  if ( ! missing( group ) & ! missing( groupVariable ) ) {
    if ( ! (groupVariable %in% names( mxdfin ) ) ) stop("group name is wrong")
    gsel = mxdfin[,groupVariable] %in% group
    if ( sum(gsel) < 5 ) stop("too few subjects in subgroup for training")
    subdf = mxdfin[ gsel, measureColumns ]
  } else {
    subdf = mxdfin[ , measureColumns ]
    group=NA
  }

  if ( ismlr3 ) {
    task = as_task_clust(subdf)
    learner = mlr_learners$get(method)
    myparams = learner$param_set$ids()
    plist = list()
    if ( "k" %in% myparams  ) plist[["k"]]=desiredk
    if ( "num_clusters" %in% myparams  ) plist[["num_clusters"]]=desiredk
    if ( "centers" %in% myparams  ) plist[["centers"]]=desiredk
    learner$param_set$values = plist
    myp = learner$train(task)
    predict(myp,newdata=subdf)
    return( myp )
  }
  subdf = data.matrix( subdf )

  if ( method == "GMM" ) {
    if ( missing( desiredk ) ) {
      if ( missing( maxk ) ) maxk = 10
#      opt_gmm = ClusterR::Optimal_Clusters_GMM(subdf, max_clusters = maxk, criterion = "BIC",
#        dist_mode = "maha_dist", seed_mode = "random_subset",
#        km_iter = 10, em_iter = 10, var_floor = 1e-10,
#        plot_data = FALSE )
#      desiredk = which.min( opt_gmm )
        opt_gmm = rep( NA, maxk )
        opt_gmm_delta = Inf
        ii = 1
        if ( trainTestRatio == 0 ) {
          isTrain = rep( TRUE, nrow(subdf) )
          isTest = rep( TRUE, nrow(subdf) )
        } else {
          isTrainNum = sample( 1:nrow( subdf ), round( trainTestRatio * nrow( subdf ) ) )
          isTrain = rep( FALSE, nrow(subdf) )
          isTrain[ isTrainNum ] = TRUE
          isTest = !isTrain
        }
        while ( opt_gmm_delta > frobNormThresh & ii < maxk ) {
          gmm = ClusterR::GMM(subdf[isTrain,], gaussian_comps = ii,
              dist_mode = "maha_dist", seed_mode = "random_subset",
              km_iter = 10, em_iter = 5, verbose = FALSE)
          gmmclp = predictSubtypeClusterMulti( subdf[isTest,], measureColumns, gmm,
              clustername = "predictSubtypeClusterMultiClustersX" )
          if ( ii == 1 ) {
            myform = as.formula( "data.matrix(subdf[isTest,]) ~  1" , env = .env)
          } else {
            myform = as.formula( "data.matrix(subdf[isTest,]) ~  factor(gmmclp$predictSubtypeClusterMultiClustersX)" , env = .env)
          }
          mdl = lm( myform )
          opt_gmm[ii] = norm( data.matrix(predict(mdl,newdata=gmmclp)) - data.matrix(subdf[isTest,]), "F")
          if ( ii > 1 ) opt_gmm_delta = 1.0 - opt_gmm[ ii ]/opt_gmm[ ii - 1 ]
          ii = ii + 1
        }
        desiredk = ii
    }
    gmm = ClusterR::GMM( subdf,
        gaussian_comps = desiredk,
        dist_mode = "maha_dist",
        seed_mode = "random_subset",
        km_iter = 10,
        em_iter = 5, verbose = FALSE )
    return( gmm )
  }
  if ( method == "kmeans" ) {
    if ( missing( desiredk ) ) {
      if ( missing( maxk ) ) maxk = 10
      opt = ClusterR::Optimal_Clusters_KMeans( subdf, 
        max_clusters = maxk, plot_clusters = FALSE,
        criterion = 'distortion_fK', fK_threshold = 0.85,
        initializer = 'optimal_init', tol_optimal_init = 0.2)
      desiredk = which.min(opt)
      }
    km_rc = # stats::kmeans( subdf, desiredk )
      ClusterR::KMeans_rcpp(subdf,
        clusters = desiredk, num_init = 5, max_iters = 100,
        initializer = 'optimal_init', verbose = F, fuzzy=TRUE)
    return( km_rc )
  }
  if ( method == "VarSelCluster" ) { 
    # res_with <- VarSelCluster(x, 2, nbcores = 2, initModel=40, crit.varsel = "BIC")
    return( VarSelLCM::VarSelCluster(  subdf, desiredk, nbcores = 4, initModel=50, crit.varsel = "AIC", vbleSelec=FALSE ) )
  }
  if ( method == "mclust" ) { 
    return( mclust::Mclust(  subdf, desiredk ) )
  }
  if ( method == "medoid" ) {
    cl_X = data.matrix( subdf )
    for (i in 1:ncol(subdf)) { cl_X[, i] = as.numeric(cl_X[, i]) }
    if ( missing( desiredk ) ) {
      opt = Optimal_Clusters_Medoids(
          cl_X,
          max_clusters=maxk,
          distance_metric=distance_metric,
          criterion = "dissimilarity",
          clara_samples = 0,
          clara_sample_size = 0,
          minkowski_p = 1,
          swap_phase = TRUE,
          threads = 1,
          verbose = FALSE,
          plot_clusters = FALSE,
          seed = 1
        )
      return(opt)
    }
    cl_f = Cluster_Medoids(cl_X, clusters = desiredk, 
      distance_metric = distance_metric, minkowski_p = 1,
       threads = 1,
       swap_phase = TRUE,
       fuzzy = TRUE,
       verbose = FALSE )
    return( cl_f )
  }
  if ( method == "pamCluster" ) return( pamCluster(subdf,desiredk) )
  if ( method == "nmfCluster" ) return( nmfCluster(subdf,desiredk) )
  if ( method == "hierarchicalCluster" ) {
    if ( is.null(distance_metric) ) distance_metric='euclidean'
    return( hierarchicalCluster(subdf,desiredk,distmethod=distance_metric) )
  }
  if ( method == "EMCluster" ) return( EMCluster(subdf,desiredk) )
  if ( method == "FuzzyCluster" ) return( FuzzyCluster(subdf,desiredk) )
  flexmeth = c("kmeansflex", "flexkmeans", "kmedians", 
    "angle", "jaccard", "ejaccard","bootclust",
    "hardcl","neuralgas", "flexcorr","cckmeans",'flexmink','flexcanb')
  if ( method %in% clustermeth ) {
    if ( method == "clara" ) return( clara(subdf, desiredk ))
    if ( method == "fanny" ) return( fanny(subdf, desiredk ))
    if ( method == "pam" ) return( pam(subdf, desiredk ))
  }
  if ( method %in% flexmeth ) {
    initk = ClusterR::KMeans_rcpp(subdf,
        clusters = desiredk, num_init = 5, max_iters = 100,
        initializer = 'optimal_init', verbose = F, fuzzy=TRUE)$clusters
    if ( method %in% c('flexcorr','flexcanb','flexmink' ) ) {
      if ( method == 'flexcorr' ) ejacFam <- flexclust::kccaFamily(dist=distCor,cent=centMean)
      if ( method == 'flexcanb' ) ejacFam <- flexclust::kccaFamily(dist=distCanberra,cent=centMean)
      if ( method == 'flexmink' ) ejacFam <- flexclust::kccaFamily(dist=distMinkowski,cent=centMean)
      mycl = flexclust::kcca(
          subdf,
          k = initk,
          weights=flexweights, group=flexgroup,
        family = ejacFam )
      return( mycl )
    }
    if ( method %in% c("hardcl","neuralgas","cckmeans"))
      if ( method == 'cckmeans' ) {
        return(
          cclust(subdf, k=initk, weights=flexweights, group=flexgroup, method='kmeans' ) )
      } else return(
        cclust(subdf, k=initk, weights=flexweights, group=flexgroup, method=method ) )
    if ( method == "bootclust ")
      return( 
        bootFlexclust(subdf, k=2:desiredk, nboot=100, FUN=cclust) )
    if ( method == "kmeansflex" | method == "flexkmeans" ) method='kmeans'
      if ( !is.null(groupFun ) ) {
        #  c( "minSumClusters", "majorityClusters", "differentClusters" )
        myfam = kccaFamily(method, trim=0.005, groupFun =groupFun)
        } else {
        myfam = kccaFamily(method, trim=0.005)
        }
    return( 
      flexclust::kcca(
          subdf,
          k = initk,
          weights=flexweights, group=flexgroup,
        family = myfam ) )
  }

}



#' Bug fix to predict function in the VarSelLCM package
#'
#' @param object varselobject
#' @param newdata newdata for prediction
#' @export
VarSelLCMproba.post <- function(object, newdata){

  logprob <- matrix(object@param@pi, nrow(newdata), object@model@g, byrow=TRUE)
  for (nom in colnames(newdata)){
    xnotna <- newdata[,which(colnames(newdata)==nom)]
    where <- which(!is.na(xnotna))
    xnotna <- xnotna[where]
    if (nom %in% rownames(object@param@paramContinuous@mu)){
      who <- which(nom == rownames(object@param@paramContinuous@mu))
      for (k in 1:object@model@g) logprob[where,k] <- logprob[where,k] + dnorm(xnotna, object@param@paramContinuous@mu[who,k], object@param@paramContinuous@sd[who,k], log=TRUE)
    }else if (nom %in% rownames(object@param@paramInteger@lambda)){
      who <- which(nom == rownames(object@param@paramInteger@lambda))
      for (k in 1:object@model@g) logprob[where,k] <- logprob[where,k] + dpois(xnotna, object@param@paramInteger@lambda[who,k], log=TRUE)
    }else if (nom %in% names(object@param@paramCategorical@alpha))
      who <- which(nom ==  names(object@param@paramCategorical@alpha))
      if ( length(object@param@paramCategorical@alpha ) > 0 )
        for (k in 1:object@model@g){
          for (h in 1:ncol(object@param@paramCategorical@alpha[[who]]))
            logprob[where,k] <- logprob[where,k] + log(object@param@paramCategorical@alpha[[who]][k,h] ** (xnotna == colnames(object@param@paramCategorical@alpha[[who]])[h]))
      }
  }
  prob <- exp(logprob - apply(logprob, 1, max))
  prob/rowSums(prob)
}

#' Reorder a subtype variable based on an external reference vector
#'
#' @param mxdfin Input data frame
#' @param clustername column name for the identified clusters
#' @param reorderingVariable reorder the cluster names based on this variable
#' @return the new dataframe and (if present) membership variables
#' @author Avants BB
#' @export
reorderingDataframe <- function( mxdfin, clustername, reorderingVariable ) {
  xdf=aggregate( mxdfin[,reorderingVariable], list(mxdfin[,clustername]), mean, na.rm=TRUE )
  neword = order(xdf[,'x'],decreasing=FALSE)
  newname = as.character( xdf[neword,'Group.1'] )
  newvarval = xdf[neword,'x']
  xdf = cbind( xdf, newname, newvarval )
  names(xdf)=c('originalname','x', 'newname', paste0("ord_x"))
  return(xdf)
}

#' Predict subtype from multivariate data
#'
#' This is the inference module for subtype definition based on a matrix.
#' Currently only supports clustering based on gaussian mixtures
#' via the ClusterR package.
#'
#' @param mxdfin Input data frame
#' @param measureColumns vector defining the data columns to be used for clustering.
#' @param clusteringObject a clustering object to predict clusters
#' @param clustername column name for the identified clusters
#' @param idvar variable name for unique subject identifier column
#' @param visitName the column name defining the visit variables
#' @param baselineVisit the string naming the baseline visit
#' @param reorderingDataframe reorder the cluster names based on this dataframe mapping of original to new variable names
#' @param distance_metric see medoid methods in ClusterR
#' @return the clusters attached to the data frame; also returns membership probabilities
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' rbfnames = names(mydf)[grep("Random",names(mydf))]
#' gmmcl = trainSubtypeClusterMulti( mydf, rbfnames, maxk=4 )
#' gmmclp = predictSubtypeClusterMulti( mydf, rbfnames, gmmcl )
#' @export
predictSubtypeClusterMulti  <- function(
  mxdfin,
  measureColumns,
  clusteringObject,
  clustername = 'GMMClusters',
  idvar,
  visitName,
  baselineVisit,
  reorderingDataframe,
  distance_metric = 'pearson_correlation'
) {
  myclustclass = class(clusteringObject)
  if ( length(myclustclass) == 1 )
    myclustclass=c(myclustclass,'other')
  subdf = mxdfin[ , measureColumns ]
  subdf = data.matrix( subdf )
  clustermeth = c("clara","fanny","pam")
  if ( myclustclass[1] %in% clustermeth ) {
    stop(paste("Prediction is not defined for ",myclustclass[1]))
  }
  if ( myclustclass[1] == "VSLCMresults" ) {
    knownnames = clusteringObject@data@var.names
    mypred = VarSelLCMproba.post( clusteringObject, subdf )
    classesare = apply( mypred, which.max, MARGIN=1 )
    mxdfin = cbind( mxdfin, factor( paste0(clustername,classesare ) ))
    colnames( mxdfin )[ ncol( mxdfin ) ] = clustername
    cluster_memberships = data.frame(mypred)
    colnames(cluster_memberships) = paste0(clustername,"_mem_",1:ncol(cluster_memberships))
    mxdfin = cbind( mxdfin, cluster_memberships ) 
  } else if ( myclustclass[2] == "Gaussian Mixture Models" ) {
    pr = ClusterR::predict_GMM( subdf, clusteringObject$centroids,
      clusteringObject$covariance_matrices, clusteringObject$weights )
    mxdfin = cbind( mxdfin, factor( paste0(clustername,pr$cluster_labels ) ))
    colnames( mxdfin )[ ncol( mxdfin ) ] = clustername
    cluster_memberships = data.frame(pr$cluster_proba)
    colnames(cluster_memberships) = paste0(clustername,"_mem_",1:nrow( clusteringObject$centroids))
    mxdfin = cbind( mxdfin, cluster_memberships )
  } else if (  myclustclass[2] == "k-means clustering" ) {
    # compute distance of every subject to each centroid
#    clusteringObject = stats::kmeans( subdf, desiredk )
    cluster_labels = rep( NA, nrow( subdf ) )
    cluster_memberships = matrix( nrow=nrow( subdf ), ncol=nrow( clusteringObject$centroids) )
    for ( kk in 1:nrow( subdf ) ) {
      dd = rep( NA, nrow( clusteringObject$centroids) )
      for ( jj in 1:nrow( clusteringObject$centroids) ) {
        dd[jj]=mean( ( as.numeric(subdf[kk,])-clusteringObject$centroids[jj,])^2 )
        cluster_memberships[kk,jj] = dd[jj]
        }
      cluster_labels[kk] = which.min(dd)
    }
    mxdfin = cbind( mxdfin, factor( paste0(clustername,cluster_labels) ) )
    colnames( mxdfin )[ ncol( mxdfin ) ] = clustername
    cluster_memberships = data.frame(cluster_memberships)
    colnames(cluster_memberships) = paste0(clustername,"_mem_",1:nrow( clusteringObject$centroids))
    mxdfin = cbind( mxdfin, cluster_memberships )
  } else if ( myclustclass[2] == "cluster medoids silhouette" ) { 
    cl_X = subdf
    for (i in 1:ncol(subdf)) { cl_X[, i] = as.numeric(cl_X[, i]) }
    pr = ClusterR::predict_Medoids( cl_X, clusteringObject$medoids,
      distance_metric =  distance_metric, fuzzy=TRUE )
    return( pr )
    cluster_labels = rep( NA, nrow( subdf ) )
    nclust = nrow(clusteringObject$medoids)
    cluster_memberships = matrix( nrow=nrow( subdf ), ncol=nclust )
    for ( kk in 1:nrow( subdf ) ) {
      dd = rep( NA, nclust )
      for ( jj in 1:nclust ) {
        dd[jj]=mean( ( as.numeric(subdf[kk,])-clusteringObject$centroids[jj,])^2 )
        cluster_memberships[kk,jj] = dd[jj]
        }
      cluster_labels[kk] = which.min(dd)
    }
    mxdfin = cbind( mxdfin, factor( paste0(clustername,cluster_labels) ) )
    colnames( mxdfin )[ ncol( mxdfin ) ] = clustername
    cluster_memberships = data.frame(cluster_memberships)
    colnames(cluster_memberships) = paste0(clustername,"_mem_",1:nrow( clusteringObject$centroids))
    mxdfin = cbind( mxdfin, cluster_memberships )
  } else if ( myclustclass[1] == "kcca" ) {
    cluster_labels = flexclust::predict( clusteringObject, newdata=subdf )
    mxdfin = cbind( mxdfin, factor( paste0(clustername,cluster_labels) ) )
    colnames( mxdfin )[ ncol( mxdfin ) ] = clustername
  } else if ( myclustclass[1] == "Mclust" ) {
    mypr = mclust::predict.Mclust( clusteringObject, newdata=subdf )
    cluster_labels = mypr$classification
    mxdfin = cbind( mxdfin, factor( paste0(clustername,cluster_labels) ) )
    colnames( mxdfin )[ ncol( mxdfin ) ] = clustername
    cluster_memberships = data.frame(mypr$z)
    colnames(cluster_memberships) = paste0(clustername,"_mem_",1:ncol(mypr$z))
    mxdfin = cbind( mxdfin, cluster_memberships )
  } else if ( myclustclass[1] %in% c("FuzzyCluster","pamCluster","EMCluster","hierarchicalCluster","nmfCluster") ) {
    mypr = predict( clusteringObject, newdata=subdf )
    cluster_labels =  paste0(clustername,as.character(mypr$classification))
    mxdfin = cbind( mxdfin, factor( cluster_labels ) )
    colnames( mxdfin )[ ncol( mxdfin ) ] = clustername
  } else if ( length(grep("Learner",myclustclass[1]))==1  ) {
    cluster_labels = predict( clusteringObject, newdata=data.frame(subdf) )
    mxdfin = cbind( mxdfin, factor( paste0(clustername,cluster_labels) ) )
    colnames( mxdfin )[ ncol( mxdfin ) ] = clustername
  } else stop("Unknown class of clustering object.")

  if ( ! missing( reorderingDataframe ) ) {
    # identify the mean value of the reo variable per class
    newclustername = rep(NA,nrow(mxdfin))
    for ( zz in 1:nrow(reorderingDataframe) ) {
      newclustername[  as.character(mxdfin[,clustername]) == as.character(reorderingDataframe[zz,'originalname']) ]=reorderingDataframe[zz,'newname']
    }
    mxdfin[,clustername]=as.character(mxdfin[,clustername])
#    mxdfin[,clustername]=factor(as.character(newclustername),levels=newclustername)
    mxdfin[,clustername]=newclustername
  }

  if ( missing( visitName ) | missing( baselineVisit ) )
    return( data.frame( mxdfin ) )

  msr = measureColumns[1]
  if ( ! ( baselineVisit %in% unique( mxdfin[,visitName] ) ) )
    stop( "! ( baselineVisit %in% unique( mxdfin[,visitName] ) )" )
  uids = unique( mxdfin[,idvar] )
  for ( u in uids ) {
    usel = mxdfin[,idvar] == u
    losel0 = mxdfin[,idvar] == u & mxdfin[,visitName] == baselineVisit & !is.na(mxdfin[,msr])
    losel0[ is.na( losel0 ) ] = FALSE
    if ( sum( losel0 ) > 0 ) {
      mytbl = table( mxdfin[losel0,clustername] )
      mostcommon = names( mytbl )[which.max(mytbl)]
      mxdfin[usel,clustername] = mostcommon
    }
  }
  return( data.frame( mxdfin ) )
}





#' Matrix factorization biclustering
#'
#' Cluster both rows and columns.  Return the joint cluster membership.
#'
#' @param mxdfin Input data frame
#' @param measureColumns vector defining the data columns to be used for clustering.
#' @param k the number of clusters
#' @param verbose boolean
#' @return a matrix with label identities; 0 indicates not in a cluster.
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 10 )
#' rbfnames = names(mydf)[grep("Random",names(mydf))]
#' mybic = biclusterMatrixFactorization( mydf, rbfnames, k = 2 )
#' @importFrom magrittr %>%
#' @importFrom ggpubr ggboxplot ggdotplot ggbarplot theme_pubr
#' @importFrom mlr3 lrn msr as_task_classif as_learner
#' @importFrom mlr3pipelines po selector_type %>>%
#' @importFrom fastICA fastICA
#' @importFrom data.table as.data.table
#' @importFrom mclust Mclust predict.Mclust mclustBIC
#' @importFrom fpc pamk
#' @importFrom flexclust kccaFamily kcca bootFlexclust cclust distCor centMean centAngle centMedian centOptim distJaccard distCanberra distAngle distEuclidean distManhattan distMax distMinkowski
#' @importFrom Evacluster pamCluster nmfCluster kmeansCluster hierarchicalCluster FuzzyCluster EMCluster 
#' @importFrom VarSelLCM VarSelCluster 
#' @importFrom Evacluster predict.pamCluster predict.nmfCluster predict.kmeansCluster predict.hierarchicalCluster predict.FuzzyCluster predict.EMCluster 
#' @export
biclusterMatrixFactorization  <- function(
  mxdfin,
  measureColumns,
  k = 2,
  verbose = FALSE
) {

  fixmat <- function( x ) {
    if ( is.null( rownames( x ) ) )
      rownames( x )=as.character( 1:nrow(x) )
    if ( is.null( colnames( x ) ) )
      colnames( x )=as.character( 1:ncol(x) )
    xt = t( x )
    zz=apply( xt, FUN=var, MARGIN=2)
    x = t( xt[, zz != 0 ] )
    zz=apply( x, FUN=var, MARGIN=2)
    myrm = rowMeans( x, na.rm=T )
    for ( rr in which( zz == 0 ) )
      x[,rr]=myrm
    return( x - min( x, na.rm=TRUE ) )
  }

  mybcmat = fixmat( data.matrix( mxdfin[,measureColumns]  ) )
  biclustmat = matrix( 0, nrow = nrow( mybcmat ), ncol = ncol( mybcmat ) )
  if ( is.null( rownames( mybcmat ) ) )
    rownames( mybcmat )=as.character( 1:nrow(mybcmat) )
  colnames( biclustmat ) = colnames( mybcmat )
  rownames( biclustmat ) = rownames( mybcmat )
  mynmf = NMF::nmf( mybcmat, k, method="Frobenius", seed='ica' )
#  w <- NMF::basis( mynmf )
#  h <- NMF::coef( mynmf )
  maxw = NMF::predict( mynmf, 'columns')
  maxh = NMF::predict( mynmf, 'rows')
  for ( j in 1:k ) {
    mycol = names( maxw )[ maxw == j ]
    myrow = names( maxh )[ maxh == j ]
    if ( verbose ) {
      print( k )
      print( mycol )
      print( myrow )
      }
    biclustmat[ rownames( biclustmat ) %in% myrow ,  colnames( biclustmat ) %in% mycol  ] = j
    }
 return( list( rowClusters=maxh, colClusters=maxw, jointClusters=biclustmat ) )
}






#' subtype feature importance
#'
#' Associate predictors (or features) with subtypes; these could be diagnoses
#' or cluster assignments.  Will use regression to return data frames intended
#' to visualize the most important relationships between features and types.
#' There are two approaches - worth using both.  Can be combined with
#' boostrapping to give distributional visualizations.
#'
#' @param dataframein Input dataframe with all relevant data
#' @param subtypeLabels Input subtype assignments.
#' @param featureMatrix matrix/dataframe defining the data columns as features.
#' @param associationType either predictor features from subtypes or predict
#' subtypes from features.  will produce related but complementary results. in
#' some cases, depending on subtypes/degrees of freedom, only one will be appropriate.
#' the third option (subjects) reports rownames of the dataframe that best fit the 
#' related subtype.
#' @param covariates optional string of covariates
#' @param transform optional effect_size
#' @param significance_level to threshold effects
#' @param visualize boolean
#' @return dataframes for visualization that show feature to subtype importance e.g. via \code{pheatmap}
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' rbfnames = names(mydf)[grep("Random",names(mydf))]
#' fimp = featureImportanceForSubtypes( mydf, mydf$DX, mydf[,rbfnames], "subtypes2features" )
#' @importFrom fastICA fastICA
#' @importFrom mclust quantileMclust
#' @importFrom coca coca
#' @importFrom cluster fanny pam clara
#' @importFrom cluster fanny pam clara
#' @importFrom effectsize t_to_d z_to_d
#' @export
featureImportanceForSubtypes <- function(
    dataframein,
    subtypeLabels,   # cluster
    featureMatrix,   # featureColumns
    associationType = c( "features2subtypes", "subtypes2features", "subjects" ),
    covariates = "1",
    transform = 'effect_sizes',
    significance_level = 0.001,
    visualize = FALSE ) {
  stopifnot( associationType %in% c( "features2subtypes", "subtypes2features", "subjects" ) )
  form_pred<-function (object, ...)
  {
      if ( is.character(object)) object = as.formula(object)
      if (inherits(object, "formula")) {
          object <- terms(object)
      }
      y_index <- attr(object, "response")
      if (y_index != 0) {
          object[[2]] <- NULL
          object <- terms(object)
      }
      all.vars(object, ...)
  }
  if ( !is.factor(subtypeLabels)  )
    subtypeLabels=factor(subtypeLabels)
  if ( length( subtypeLabels ) != nrow( featureMatrix ) )
    stop("length( subtypeLabels ) != nrow( featureMatrix )")
  uniqClusts = levels( subtypeLabels )
  mync = length( uniqClusts )
  # stores feature importance
  clustzdescribe = data.frame( matrix( nrow = mync, ncol = ncol( featureMatrix ) ) )
  clustsigdescribe = data.frame( matrix( nrow = mync, ncol = ncol( featureMatrix ) ) )
  colnames( clustsigdescribe ) = colnames( featureMatrix )
  rownames( clustsigdescribe ) = uniqClusts
  colnames(clustzdescribe) = colnames( featureMatrix )
  clustmat = matrix( 0, nrow = length( subtypeLabels ), ncol=mync )
  colnames(clustmat) = as.character( uniqClusts )
  for ( j in 1:mync ) {
    losel = subtypeLabels == uniqClusts[j]
    if ( sum(losel) > 0 )
      clustmat[ losel , j] = 1
    }
  if ( associationType[1] == "subjects" ) {
    bestSubjectList = list()
    for ( j in 1:mync ) {
        mygt = gt( clustmat[,j],data.matrix(featureMatrix), 
          permutations=round(1.0/significance_level), standardize=TRUE, 
          model='logistic' )
        mysub = data.frame( subjects( mygt, what="z-score", sort=TRUE, cluster=FALSE ) )
        mysub[,'zscore']=( mysub[,'Residual'] ) / mysub[,'Std.dev']
        bestSubjectList[[j]]=mysub
        }
    return( bestSubjectList )
  } else if ( associationType[1] == "features2subtypes" ) {
    # converts cluster labels to one-hot coding
    for ( j in 1:mync ) {
      # return( list(clustmat,featureMatrix))
      myrform = paste(" temp ~ ", covariates )
      featureMatrixResid=featureMatrix
      for ( kk in 1:ncol(featureMatrix) ) {
        dataframein$temp = featureMatrix[,kk]
        featureMatrixResid[,kk] = residuals( lm( myrform, data=dataframein ) )
      }
      mygt = gt( clustmat[,j],data.matrix(featureMatrixResid), 
        permutations=round(1.0/significance_level), standardize=TRUE, 
        model='logistic'  )
      mycov = covariates( mygt, what="z-score", zoom=TRUE, cluster=FALSE, plot=FALSE )
      mycov = extract( mycov )
      mycoffs = data.frame( cbind( mycov@result, mycov@extra ) )
      mycoffssub=mycoffs[ mycoffs$direction == "assoc. with clustmat[, j] = 1" & 
        mycoffs$p.value <= 0.05, ]
      myz = mycoffs[,"Statistic"]
      myzsub = mycoffssub[,"Statistic"]
      if ( transform == 'effect_sizes' ) {
        myz = as.numeric( effectsize::z_to_d( myz, nrow(featureMatrix) )$d )
        if ( nrow( mycoffssub ) > 0 ) {
          myzsub = as.numeric( effectsize::z_to_d( myzsub, nrow(featureMatrix))$d  )
        } else myzsub=myz
        }
      clustzdescribe[j,rownames(mycoffs)]=myz
      clustsigdescribe[j,rownames(mycoffssub)]=myzsub
    }
    if ( visualize ) pheatmap::pheatmap(abs(clustzdescribe),cluster_rows=F,cluster_cols=F)
    # get the max for each column
    clustsigdescribemax = clustsigdescribe * 0
    for ( j in 1:ncol(clustzdescribe) ) {
      wmax = which.max(abs(clustzdescribe[,j]))
      clustsigdescribemax[ wmax, j ] = clustzdescribe[wmax,j]
    }
    if ( visualize ) pheatmap::pheatmap((clustsigdescribemax),cluster_rows=F,cluster_cols=F)
    return( list(
      subtypeFeatureZScores = clustzdescribe,
      subtypeFeatureZScoresSignificant = clustsigdescribe,
      subtypeFeatureZScoresMax = clustsigdescribemax
    ) )
  } else {
    myform = as.formula(paste( "featureMatrix[,j] ~ clustmat[,k] + ", covariates ))
    if ( covariates != "1")
      locvars = form_pred( myform )[-c(1:2)]
    for ( j in 1:ncol(featureMatrix) ) {
      for( k in 1:mync ) {
        localdf = data.frame(
          feat=featureMatrix[,j],
          clust=clustmat[,k])
        if ( covariates != "1") {
          temper = dataframein[,locvars]
          if (length(locvars)==1) {
            temper=data.frame( temper )
            colnames(temper)=locvars
            }
          localdf=cbind(localdf, temper)
        }
        c1reg = lm( as.formula(myform), data=localdf )
        mycoffs = coefficients(summary(c1reg))
        if ( nrow(mycoffs ) > 1 ) {
          sigthresh = as.numeric( mycoffs[2,"Pr(>|t|)"] <= significance_level )
          myz = mycoffs[2,"t value"]
          if ( transform == 'effect_sizes' ) 
            myz = data.frame(effectsize::t_to_d( myz, nrow(localdf) ))[1,1]
          clustzdescribe[k,j]=myz
          clustsigdescribe[k,j]=myz * sigthresh
        } else {
          clustzdescribe[k,j]=NA
          clustsigdescribe[k,j]=NA
        }
      }
    }
    # get the max for each column
    clustsigdescribemax = clustsigdescribe * 0
    for ( j in 1:nrow(clustzdescribe) ) {
      wmax = which.max(abs(clustzdescribe[j,]))
      clustsigdescribemax[ j, wmax ] = clustzdescribe[j, wmax]
    }
    if ( visualize ) pheatmap::pheatmap((clustsigdescribemax),cluster_rows=F,cluster_cols=F)
    rownames(clustzdescribe)=rownames(clustsigdescribemax)
    return( list(
      subtypeFeatureTScores = clustzdescribe,
      subtypeFeatureTScoresSignificant = clustsigdescribe,
      subtypeFeatureTScoresMax = clustsigdescribemax
    ) )

  }
}



#' select top features via linear regression on an outcome
#'
#' @param dataframein Input dataframe with all relevant data
#' @param subtypeLabels Input subtype assignments.
#' @param featureMatrix matrix/dataframe defining the data columns as features.
#' @param covariates optional string of covariates
#' @param n_features select this many features per level
#' @param associationType either predictor features from subtypes or predict
#' subtypes from features.  will produce related but complementary results. in
#' some cases, depending on subtypes/degrees of freedom, only one will be appropriate.
#' @return vector of feature names
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' rbfnames = names(mydf)[grep("Random",names(mydf))]
#' fimp = regressionBasedFeatureSelection( mydf, mydf$DX, mydf[,rbfnames], 
#'    associationType ="subtypes2features" )
#' @export
regressionBasedFeatureSelection <- function( 
    dataframein, subtypeLabels, featureMatrix, covariates="1", n_features=25, associationType ="features2subtypes" ) {
    fimp = featureImportanceForSubtypes(
        dataframein,
        subtypeLabels, 
        featureMatrix, 
        associationType[1], 
        covariates=covariates )
    if ( "subtypeFeatureTScoresSignificant" %in% names(fimp) ) {
      fimpname = "subtypeFeatureTScoresSignificant"
    } else if ( "subtypeFeatureZScoresSignificant" %in% names(fimp) ) {
      fimpname = "subtypeFeatureZScoresSignificant"
    }
    thefeats = c()
    myfimp = fimp[[fimpname]]
    for ( k in 1:nrow( myfimp )) {
        mm = abs(myfimp[k,])
        myor = order( as.numeric(mm[1,]), decreasing=TRUE )
        thefeats = c( thefeats, colnames( mm )[ head(myor,n_features) ] )
        }
    return( unique( thefeats ) )
    }





#' subtype plotting
#'
#' Assemble a set of standard plots looking at subtype results to support comparing
#' across a hierarchy of types both cross-sectionally and longitudinally.
#'
#' @param inputDataFrame Input complete data frame
#' @param variableToVisualize string naming the variable to display across subtypes
#' @param hierarchyOfSubtypes string vector of subtypes with increasing degrees of specificity
#' @param idvar variable name for unique subject identifier column
#' @param vizname the name of the grouped time variable (e.g. years change rounded to nearest quarter year)
#' @param whiskervar character either ci or se
#' @param consistentSubset display longitudinal data only from subjects that are consistently present at all visits
#' @param manualColors a list of user defined manual colors; the length of this list
#' should match the length of hierarchyOfSubtypes and colors should be named according
#' to the levels therein.  each entry in the list should be a string vector of color names.
#' @param outputPrefix filename prefix for the stored pdf plots; if missing, just plot to display
#' @param width the width of the graphics region in inches.
#' @param height the height of the graphics region in inches.
#' @return the output is a set of plots saved at the outputPrefix location
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 1000 )
#' qdf = trainSubtypeUni( mydf, "cognition", c("C0","C1","C2"), c(0.33,0.66) )
#' qdf = predictSubtypeUni( mydf, qdf, "Id" )
#' \dontrun{
#' hierarchicalSubtypePlots( qdf, "cognition", c("DX", "subtype" ),
#'  "Id", "visit", outputPrefix='/tmp/X' )
#' }
#' @importFrom globaltest gt covariates extract subjects
#' @importFrom ggstatsplot ggbetweenstats
#' @importFrom wesanderson wes_palette wes_palettes
#' @importFrom ggthemes theme_tufte
#' @importFrom magrittr %>%
#' @importFrom ggbeeswarm geom_beeswarm
#' @importFrom dplyr sym left_join mutate summarize n group_by
#' @importFrom ggplot2 labs element_text theme geom_smooth geom_violin geom_boxplot geom_dotplot ggtitle stat_smooth
#' @importFrom ggplot2 facet_grid facet_wrap label_both vars
#' @export
hierarchicalSubtypePlots <- function(
    inputDataFrame,
    variableToVisualize,
    hierarchyOfSubtypes,
    idvar,
    vizname,
    whiskervar=c('ci','se'),
    consistentSubset = FALSE,
    manualColors,
    outputPrefix,
    width=12, height=8 ) {
  if ( ! ( any(hierarchyOfSubtypes %in% names(inputDataFrame) ) ) )
    stop("some hierarchyOfSubtypes variables do not exist in the data frame.")
  if ( ! ( any(variableToVisualize %in% names(inputDataFrame) ) ) )
    stop(paste("the variableToVisualize", variableToVisualize, "does not exist in the data frame."))
  if ( ! missing( manualColors ) )
    stopifnot( length(manualColors) == length( hierarchyOfSubtypes ) )
  figs = c()
  ct = 1
  if ( missing( outputPrefix ) ) visualize = TRUE else visualize = FALSE

  # first level in hierarchy
  for (  k in 1:length( hierarchyOfSubtypes ) ) {
    losel = !is.na(inputDataFrame[,hierarchyOfSubtypes[k]])
    losel[ is.na(losel) ] = FALSE
    xplot0 <- ggstatsplot::ggbetweenstats(
          data = inputDataFrame[losel,],
          x = !!dplyr::sym( hierarchyOfSubtypes[k] ),
          y = !!dplyr::sym( variableToVisualize ),
          plot.type = "boxviolin", # type of plot
          bf.message = FALSE,
          results.subtitle = FALSE,
#          ggtheme = ggthemes::theme_tufte(), # ggplot2::theme_minimal(), # a different theme
          package = "wesanderson",
          palette = "Royal2"
        ) + ggplot2::labs( x = hierarchyOfSubtypes[k], y = variableToVisualize,
        title = paste("Subtype:",hierarchyOfSubtypes[k], "vs", variableToVisualize ) ) +
        ggplot2::theme(text = ggplot2::element_text(size = 20) ) +
        ggplot2::theme(plot.title = ggplot2::element_text(size=22))
    if ( ! missing( manualColors ) )
      xplot0 <- xplot0 + scale_colour_manual(values = manualColors[[k]] )
    if ( visualize ) print( xplot0 ) else {
      poster = paste0( hierarchyOfSubtypes[k], "_vs_", variableToVisualize, ".pdf" )
      outfn = paste0( outputPrefix, "_", poster )
      pdf( outfn, width=width, height=height )
      print( xplot0 )
      dev.off()
      figs[ct] = outfn
      ct = ct + 1
      }
    }

  # 2nd level in hierarchy - do level 1 vs other levels
  # e.g. if 1st level is DX, then plot subtypes within each DX
  if ( is.factor( inputDataFrame[,hierarchyOfSubtypes[1]] ) ) {
    unqDX = levels( inputDataFrame[,hierarchyOfSubtypes[1]] )
  } else unqDX = sort( unique( inputDataFrame[,hierarchyOfSubtypes[1]] ) )
  if ( length( hierarchyOfSubtypes ) > 1 ) {
    for (  k in 2:length( hierarchyOfSubtypes ) ) {
      for ( j in 1:length( unqDX ) ) {
        losel = inputDataFrame[,hierarchyOfSubtypes[1]] == unqDX[j] &
          !is.na(inputDataFrame[,hierarchyOfSubtypes[k]])
        losel[ is.na(losel) ] = FALSE
        xplot1 <- ggstatsplot::ggbetweenstats(
              data = inputDataFrame[ losel, ],
              x = !!dplyr::sym( hierarchyOfSubtypes[k] ),
              y = !!dplyr::sym( variableToVisualize ),
              plot.type = "boxviolin", # type of plot
              bf.message = FALSE,
              results.subtitle = FALSE,
#              ggtheme = ggthemes::theme_tufte(), # ggplot2::theme_minimal(), # a different theme
              package = "wesanderson",
              palette = "Royal2"
            ) + ggplot2::labs( x = hierarchyOfSubtypes[k], y = variableToVisualize,
            title = paste("Subtype:",hierarchyOfSubtypes[k], "vs", variableToVisualize, "@", unqDX[j] ) ) +
            ggplot2::theme(text = ggplot2::element_text(size = 20) ) +
            ggplot2::theme(plot.title = ggplot2::element_text(size=22))
        if ( ! missing( manualColors ) )
          xplot1 <- xplot1 + scale_colour_manual(values = manualColors[[k]] )
        if ( visualize ) print( xplot1 ) else {
          poster = paste0( hierarchyOfSubtypes[k], "_vs_", variableToVisualize, "_at_", toString(unqDX[j]),".pdf" )
          outfn = paste0( outputPrefix, "_", poster )
          pdf( outfn, width=width, height=height )
          print( xplot1 )
          dev.off()
          figs[ct] = outfn
          ct = ct + 1
          }
        }
      }
  }

  # for facetting
  wrap_by <- function(...) {
        facet_wrap(vars(...), labeller = label_both)
      }

  # now do something similar with longitudinal data
  if ( ! missing( vizname ) ) {
    uviz = sort( unique(  inputDataFrame[,vizname] ) )
    if ( length( uviz ) == 1 ) {
#      message("length( uviz ) == 1")
#      return( figs )
    }
    mysubs = unique( inputDataFrame[,idvar] )
    if ( consistentSubset ) {
      mysubs = unique( inputDataFrame[ inputDataFrame[,vizname] == uviz[1],idvar] )
      for ( jj in 2:length(uviz) ) {
        mysubs = intersect( mysubs,
          unique( inputDataFrame[ inputDataFrame[,vizname] == uviz[jj],idvar] ) )
        }
#      print( paste( "length(mysubs) = ", length(mysubs)) )
      }
    # do first level plots
    for (  k in 1:length( hierarchyOfSubtypes ) ) {
      myxlab = paste( hierarchyOfSubtypes[k], "vs", variableToVisualize,
        "longitudinal" )
      losel = !is.na(inputDataFrame[,hierarchyOfSubtypes[k]]) &
        inputDataFrame[,idvar] %in% mysubs
      losel[ is.na(losel) ] = FALSE
      lplot0 = plotSubtypeChange(
        inputDataFrame[losel,],
        idvar = idvar,
        measurement = variableToVisualize,
        subtype = hierarchyOfSubtypes[k],
        vizname = vizname, xlab = myxlab,
        whiskervar = whiskervar[1] )
      if ( ! missing( manualColors ) )
        lplot0 <- lplot0 + scale_colour_manual(values = manualColors[[k]] )
      if ( visualize ) print( lplot0 ) else {
        poster = paste0( hierarchyOfSubtypes[k], "_vs_", variableToVisualize, "_longitudinal.pdf" )
        outfn = paste0( outputPrefix, "_", poster )
        pdf( outfn, width=width, height=height )
        print( lplot0 )
        dev.off()
        figs[ct] = outfn
        ct = ct + 1
        }



      sym2 = sym(hierarchyOfSubtypes[k])
      ttt = inputDataFrame[losel,]
      viznameB=paste0(vizname,'_c')
      ttt[,viznameB] = as.numeric( as.factor( ttt[,vizname] ) )
      lplot1 =  ggplot( ttt,
            aes(
              y=!!sym(variableToVisualize), x=!!sym(viznameB),
                fill = (!!sym2), color =  (!!sym2)
            ) ) + wrap_by(!!sym2) +
                theme_bw() +
                geom_beeswarm() +
                geom_smooth(method = "lm", alpha = .15, aes(fill = (!!sym2)) ) +
                ggtitle( myxlab ) +
                theme(text = element_text(size=20),legend.position='top')
        if ( ! missing( manualColors ) )
          lplot1 <- lplot1 + scale_colour_manual(values = manualColors[[k]] )
        if ( visualize ) print( lplot1 ) else {
          poster = paste0( hierarchyOfSubtypes[k], "_vs_", variableToVisualize,
            "_longitudinal_swarmplot.pdf" )
          outfn = paste0( outputPrefix, "_", poster )
          pdf( outfn, width=width, height=height )
          print( lplot1 )
          dev.off()
          figs[ct] = outfn
          ct = ct + 1
          }
      }


    if ( length( hierarchyOfSubtypes ) > 1 ) {
      for (  k in 2:length( hierarchyOfSubtypes ) ) {
        for ( j in 1:length( unqDX ) ) {
          myxlab = paste( hierarchyOfSubtypes[k], "vs", variableToVisualize,
            "at", toString(unqDX[j]),
            "longitudinal" )
          losel = inputDataFrame[,hierarchyOfSubtypes[1]] == unqDX[j] &
            !is.na(inputDataFrame[,hierarchyOfSubtypes[k]]) &
              inputDataFrame[,idvar] %in% mysubs
          losel[ is.na(losel) ] = FALSE
          lplot1 = plotSubtypeChange(
            inputDataFrame[losel,],
            idvar = idvar,
            measurement = variableToVisualize,
            subtype = hierarchyOfSubtypes[k],
            vizname = vizname,
            whiskervar = whiskervar[1],
            xlab = myxlab )
          if ( ! missing( manualColors ) )
            lplot1 <- lplot1 + scale_colour_manual(values = manualColors[[k]] )
          if ( visualize ) print( lplot1 ) else {
            poster = paste0( hierarchyOfSubtypes[k], "_vs_", variableToVisualize,
              "_at_", toString(unqDX[j]),
              "_longitudinal.pdf" )
            outfn = paste0( outputPrefix, "_", poster )
            pdf( outfn, width=width, height=height )
            print( lplot1 )
            dev.off()
            figs[ct] = outfn
            ct = ct + 1
            }

          sym2 = sym(hierarchyOfSubtypes[k])
          sample_size = inputDataFrame[losel,] %>% group_by(!!sym(vizname)) %>% summarize(num=n())
          ttt = inputDataFrame[losel,]
          viznameB=paste0(vizname,'_c')
          ttt[,viznameB] = as.numeric( as.factor( ttt[,vizname] ) )
          lplot1 =  ggplot( ttt,
              aes(
                y=!!sym(variableToVisualize), x=!!sym(viznameB),
                  fill = (!!sym2), color =  (!!sym2)
#                  group=interaction(!!sym(viznameB), !!sym2)
              ) ) + wrap_by(!!sym2) +
                  theme_bw() +
                  geom_beeswarm() + # stat_smooth(method='lm')
                  geom_smooth(method = "lm", alpha = .15, aes(fill = (!!sym2)) ) +
                  ggtitle( myxlab ) +
                  theme(text = element_text(size=20),legend.position='top')
          if ( ! missing( manualColors ) )
            lplot1 <- lplot1 + scale_colour_manual(values = manualColors[[k]] )
          if ( visualize ) print( lplot1 ) else {
            poster = paste0( hierarchyOfSubtypes[k], "_vs_", variableToVisualize,
              "_at_", toString(unqDX[j]),
              "_longitudinal_swarmplot.pdf" )
            outfn = paste0( outputPrefix, "_", poster )
            pdf( outfn, width=width, height=height )
            print( lplot1 )
            dev.off()
            figs[ct] = outfn
            ct = ct + 1
            }
          }
        }
      }


    }
  return( figs )

}




#' genetic variants data frame from plink data
#'
#' @param rootFileName root for pgen psam and pvar files
#' @param targetSNPs snps to extract (optional - if absent, will get all)
#' @param verbose boolean
#' @return dataframes with both variants and subject ids
#' @author Avants BB
#' @examples
#' # mydf = plinkVariantsDataFrame( fn, c( 'rs6469804', 'rs6859' ) )
#' @importFrom pgenlibr NewPvar NewPgen ReadList
#' @importFrom data.table fread
#' @importFrom gaston as.matrix read.bed.matrix
#' @export
plinkVariantsDataFrame <- function( rootFileName, targetSNPs,  verbose=FALSE ) {
  type=getExt(rootFileName)
  stopifnot( type %in% c("pgen","bed") )
  if ( type == 'pgen' ) {
    rootFileName2 = tools::file_path_sans_ext( rootFileName )
    f.pvar = paste0(rootFileName2, '.pvar')
    f.pgen = paste0(rootFileName2, '.pgen')
    f.sam = paste0(rootFileName2, '.psam')
    subjectIDs = fread( f.sam )
    myvariants = fread( f.pvar, select = 3)
    if ( missing(targetSNPs) ) i = 1:length(myvariants$ID)
    if ( !missing(targetSNPs) ) i  <- which( myvariants$ID %in% targetSNPs)
    pvar <- pgenlibr::NewPvar(f.pvar)
    pgen <- pgenlibr::NewPgen(f.pgen, pvar=pvar) #,sample_subset = c(1,2,3,4) )
    myderk = pgenlibr::ReadList( pgen, i, meanimpute=F )
    snpdf = data.frame( myderk )
    colnames( snpdf ) = myvariants$ID[i]
    outdf = data.frame( subjectIDs = subjectIDs$IID, snpdf )
    return( outdf )
  } else {
    gwas=gaston::read.bed.matrix( rootFileName )
    ww=which( gwas@snps$id %in% targetSNPs )
    if ( length( ww ) == 0 ) {
      mymsg = paste("No target SNPs are present ")
      print(mymsg)
      message(mymsg)
      return(NA)
      }
    y=gwas[,ww]
    snpnames=y@snps$id
    y=gaston::as.matrix( y )
    gmydf=data.frame(subjectIDs=rownames(y))
    gmydf=cbind(gmydf,y)
    return(gmydf)
  }
}




#' three way interaction plot from raw data
#'
#' @param indf dataframe with relevant variables
#' @param xvar variable for x-axis of plots
#' @param yvar variable for y-axis of plots
#' @param colorvar variable by which to color plots
#' @param anat continuous variable by which to split plots
#' @param anatshow optional character name for continuous variable to show on plots - can be up to vector length 3 - one for each panel of the plot
#' @param ggpalette optional palette
#' @param showpoints boolean
#' @return the plot
#' @author Avants BB
#' @examples
#' # FIXME
#' @importFrom ggpubr ggscatter set_palette
#' @importFrom gridExtra grid.arrange
#' @export
threewayinteraction <- function( indf, xvar, yvar, colorvar, anat, anatshow, ggpalette='jco', showpoints=FALSE ) {
  glist = list()
  if ( ! missing(anatshow) ) {
    while ( length(anatshow) <  3 )
      anatshow=c(anatshow,"")
  }
  if ( missing(anatshow) ) {
    anatshow=gsub("T1Hier_","",anat)
    anatshow=rep(anatshow,3)
  }
  indf[,colorvar]=factor(indf[,colorvar])
  indf$snapfact=factor(indf[,colorvar])
  ylimmer = range( indf[,yvar])
  glist[[length(glist)+1]]= ggscatter(indf, x = xvar, y = yvar, color=colorvar,   size=3.45,  point=showpoints, add = "reg.line", palette=ggpalette, conf.int=T, cor.coef=TRUE ) + theme(text = element_text(size=12))+ ylim( ylimmer )
  # ggtitle(paste(anatshow[1])) + ylim( ylimmer ) #+ theme(legend.position = "none")

  if ( is.numeric(indf[,anat]) ) {
    medsplit = median( indf[,anat], na.rm=T )
    hisel = indf[,anat] > medsplit
    loclev=loclev2=anat
  } else {
    loclev = (unique(indf[,anat])[1])
    hisel=indf[,anat]==loclev
    loclev2=paste0("!",loclev)
  }
  glist[[length(glist)+1]]=ggscatter(indf[hisel,], x = xvar, y = yvar, color=colorvar,   size=3.45, point=showpoints, add = "reg.line", palette=ggpalette, conf.int=T, cor.coef=TRUE ) + theme(text = element_text(size=12))+ ggtitle(paste('+',anatshow[2])) + theme(legend.position = "none") + ylim( ylimmer )

  
  glist[[length(glist)+1]]=ggscatter(indf[!hisel,], x = xvar, y = yvar, color=colorvar,   size=3.45, point=showpoints, add = "reg.line", palette=ggpalette, conf.int=T, cor.coef=TRUE  ) + theme(text = element_text(size=12))+ ggtitle(paste('-',anatshow[3]))+ theme(legend.position = "none") + ylim(  ylimmer )

  grid.arrange(grobs=glist,ncol=3,top=anatshow[1])

}




#' Get the file extension from a file-name. from `reader` package.
#'
#' @param fn filename(s) (with full path is ok too)
#' @return returns the (usually) 3 character file extension of a filename
#' @export 
#' @author Nicholas Cooper \email{nick.cooper@@cimr.cam.ac.uk}
getExt <- function (fn) 
{
    if (length(fn) < 1) {
        warning("fn had length of zero")
        return(fn)
    }
    if (all(is.na(fn)) | !is.character(fn)) {
        stop("fn should not be NA and should be of type character()")
    }
    strip.file.frags <- function(X) {
        file.segs <- strsplit(X, ".", fixed = TRUE)[[1]]
        lss <- length(file.segs)
        if (lss > 1) {
            out <- paste(file.segs[lss])
        }
        else {
            out <- ""
        }
        return(out)
    }
    return(sapply(fn, strip.file.frags))
}





#' Match a pair of vector distributions based on quantiles 
#'
#' @param vecToBeTransformed input vector to be transformed to match the reference
#' @param vecReference reference vector
#' @param quantiles a vector of quantile points to match
#' @param polynomialOrder integer greater than or equal to one
#' @param truncate boolean
#' @return the transformed vector
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' rbfnames = names(mydf)[grep("Random",names(mydf))]
#' newvec = matchVectorDistributionByQuantiles( mydf[,rbfnames[1]], mydf[,rbfnames[1]] )
#' @export
matchVectorDistributionByQuantiles  <- function(
  vecToBeTransformed,
  vecReference,
  quantiles = 1:9/10.0, 
  polynomialOrder = 1,
  truncate = TRUE ) {
    myq1 = quantile( vecToBeTransformed, quantiles, na.rm=T )
    myq2 = quantile( vecReference, quantiles, na.rm=T )
    mytd = data.frame( myq2=myq2, myq1=myq1 )
    mdl = lm( myq2 ~ stats::poly(myq1,polynomialOrder), data=mytd )
    mynd=data.frame( myq1 = vecToBeTransformed )
    outvec = predict( mdl, newdata=mynd  )
    if ( truncate ) {
      outvec[ outvec < min(vecReference,na.rm=T)]=min(vecReference,na.rm=T)
      outvec[ outvec > max(vecReference,na.rm=T)]=max(vecReference,na.rm=T)
    }
    return( outvec )
  }






#' quantify by quantiles (quantSquared)
#' 
#' transform input vector to a quantile representation relative to a reference. 
#'
#' @param vecToBeTransformed Input vector should have same entries as reference;
#' should be a row of a data frame
#' @param matrixReferenceDistribution data frame defining the reference data;  
#' ideally this will not contain missing data. 
#' @return the quantile transformed vector
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' rbfnames = names(mydf)[grep("Random",names(mydf))]
#' zz=data.frame(t(colMeans(mydf[,rbfnames])))
#' newvec = quantSquared( zz, mydf )
#' newvec2 = quantSquared( mydf[1,rbfnames], mydf )
#' @export
quantSquared  <- function(
  vecToBeTransformed,
  matrixReferenceDistribution ) {
    mycnt = colnames( vecToBeTransformed )
    newvec = rep( NA, length( vecToBeTransformed ) )
    names(newvec)=mycnt
    for ( x in mycnt ) {
      refvec = matrixReferenceDistribution[,x]
      nna = sum( !is.na( refvec ) )
      boolvec =  as.numeric(vecToBeTransformed[x]) > as.numeric(refvec)
      newvec[x] = sum( boolvec , na.rm=TRUE )/ nna
    }
    return( newvec )
  }

#' Augment a Data Frame with Custom Color Column
#'
#' This function adds a new column named 'custom_color' to a given data frame.
#' The new column maps factor levels of a specified column to colors
#' from a user-defined palette.
#'
#' @param df A data frame to be augmented.
#' @param column_name The name of the column in `df` whose factor levels are to be 
#'   mapped to colors. This column should exist in `df` and should be a factor or
#'   convertible to a factor.
#' @param color_palette A vector of colors, corresponding to the factor levels of
#'   the specified column. The number of colors must match the number of unique
#'   factor levels in the column.
#' @param color_palette_name optional named color palette
#' @return A data frame identical to `df` but with an additional column 
#'   'custom_color', which contains the color mappings.
#'
#' @examples
#' data <- data.frame(
#'   category = c("A", "B", "C", "A", "B"),
#'   value = c(10, 20, 30, 40, 50)
#' )
#' palette <- c("red", "green", "blue")
#' augmented_data <- augment_with_custom_color(data, "category", palette)
#'
#' @importFrom ggpubr get_palette
#' @importFrom stats setNames
#' @export
augment_with_custom_color <- function(df, column_name, color_palette, color_palette_name = NULL ) {
  # Check if the column exists in the data frame
  if (!column_name %in% names(df)) {
    stop("Column not found in the data frame")
  }

  # Convert the column to a factor if it's not already
  factor_col <- factor(df[[column_name]])

  # Extract the factor levels of the column
  factor_levels <- levels(factor_col)

  if ( ! missing( color_palette_name ) ) {
    color_palette = get_palette(palette = color_palette_name, length(factor_levels))
  }

  # Check if the number of colors matches the number of factor levels
  if (length(color_palette) != length(factor_levels)) {
    stop("The number of colors must match the number of factor levels")
  }

  # Create a named vector for color mapping
  color_mapping <- setNames(color_palette, factor_levels)

  # Map the factor levels to the colors
  df$custom_color <- color_mapping[factor_col]

  return(df)
}


#' prplot
#' 
#' Partial residual regression plot using ggpubr for display 
#' and visreg for partial residual calculation 
#'
#' @param mdl input fitted model
#' @param xvariable a model term
#' @param byvariable a model term interacting with the xvariable
#' @param titlestring string for title
#' @param ystring string for title
#' @param addpoints continuous value greater than zero
#' @param palette string
#' @param colorvar string
#' @param extradata dataframe with additional information to be rendered
#' @return the quantile transformed vector
#' @export
prplot <- function(
  mdl, xvariable, byvariable, titlestring = '', ystring = '', 
  addpoints = 0, palette = 'npg', colorvar = '', extradata = NULL
) {
  addthepoints <- FALSE
  colorvarnotempty <- TRUE
  if (colorvar == '') {
    colorvarnotempty <- FALSE
    colorvar <- 'black'
  }
  if (!is.null(extradata)) {
    extradata <- extradata[names(predict(mdl)), ]
  }
  if (addpoints > 0) addthepoints <- TRUE
  
  if (!missing(byvariable)) {
    vv <- visreg::visreg(mdl, xvariable, by = byvariable, plot = FALSE)
    
    # Ensure all levels of byvariable are preserved
    vv$res[, byvariable] <- factor(vv$res[, byvariable], levels = levels(model.frame(mdl)[, byvariable]))
    
    # Handle colorvar levels in the data
    if (colorvar %in% names(model.frame(mdl))) {
      vv$res[, colorvar] <- model.frame(mdl)[names(predict(mdl)), colorvar]
    } else if (!is.null(extradata)) {
      if (colorvar %in% colnames(extradata)) {
        vv$res[, colorvar] <- factor(extradata[, colorvar], levels = levels(model.frame(mdl)[, colorvar]))
      }
    }
    
    if (is.factor(vv$res[, xvariable]) | is.character(vv$res[, xvariable])) {
      return(
        ggdotplot(
          vv$res, x = xvariable, y = 'visregRes', size = addpoints, palette = palette,
          conf.int = TRUE, point = addthepoints, facet.by = byvariable, cor.coef = TRUE
        ) + theme(text = element_text(size = 12)) + ylab(ystring) + ggtitle(titlestring)
      )
    } else {
      myaddparams <- list(color = "blue", fill = "cyan")
      mypp <- predict(mdl)
      mylims <- range(mypp) * c(1.3, 1.0)
      return(
        ggscatter(
          vv$res, x = xvariable, y = 'visregRes', size = addpoints, palette = palette,
          point = addthepoints, add = 'reg.line', conf.int = TRUE, color = colorvar, facet.by = byvariable,
          add.params = myaddparams, cor.coef = TRUE
        ) + theme(text = element_text(size = 12)) + ylab(ystring) + ggtitle(titlestring) +
          theme(legend.position = "top", legend.title = element_blank())
      )
    }
  }
  
  if (missing(byvariable)) {
    vv <- visreg::visreg(mdl, xvariable, plot = FALSE)
    
    # Ensure colorvar levels are handled correctly
    if (colorvar %in% names(model.frame(mdl))) {
      vv$res[, colorvar] <- model.frame(mdl)[names(predict(mdl)), colorvar]
    } else if (!is.null(extradata)) {
      if (colorvar %in% colnames(extradata)) {
        vv$res[, colorvar] <- factor(extradata[, colorvar], levels = levels(model.frame(mdl)[, colorvar]))
      }
    }
    
    if (is.factor(vv$res[, xvariable]) | is.character(vv$res[, xvariable])) {
      return(
        ggboxplot(
          vv$res, x = xvariable, y = 'visregRes', size = addpoints, palette = palette,
          conf.int = TRUE, point = addthepoints, add.params = list(color = "blue", fill = "cyan"),
          fill = colorvar, cor.coef = TRUE
        ) + theme(text = element_text(size = 12)) + ylab(ystring) + ggtitle(titlestring)
      )
    } else {
      return(
        ggscatter(
          vv$res, x = xvariable, y = 'visregRes', size = addpoints, point = addthepoints,
          add = 'reg.line', conf.int = TRUE, color = colorvar, add.params = list(color = "blue", fill = "cyan"),
          palette = palette, cor.coef = TRUE
        ) + theme(text = element_text(size = 12)) + ylab(ystring) + ggtitle(titlestring) +
          theme(legend.position = "top", legend.title = element_blank())
      )
    }
  }
}

#' balanced sampling of a variable
#' 
#' resample a dataframe to counteract variable imbalance
#' 
#' @param x data frame to extend
#' @param variable column name
#' @param method one of permissible methods listed in function (will print if wrong method passed)
#' 
#' @return new data frame
#' @author Avants BB
#' @export
balancedDataframe <- function( x, variable, method ) {
    valbal = c("none", "rwo", "racog", "mwmote", "over", "under" )
    if ( ! ( method %in% valbal ) )
        stop(paste("method must be one of ", paste(valbal, collapse=" / ")))
    if ( method == 'none' ) return( x )
    temptbl = table( x[,variable] )
    myinstsize = max( temptbl ) - min( temptbl )
    if ( method == 'over') {
        return( oversample_minority( x, variable ) )
    }
    if ( method == 'under') {
        return( undersample_majority( x, variable ) )
    }
    if ( method == 'mwmote')
        balDF = mwmote( dataset = x, numInstances = myinstsize, classAttr = variable)
    if ( method == 'rwo')
        balDF = rwo( dataset = x, numInstances = myinstsize, classAttr = variable)
    if ( method == 'racog')
        balDF = racog( dataset = x, numInstances = myinstsize, classAttr = variable)
    x = rbind( x, balDF )
    return(x)
    }



#' balanced sampling of a multi-class variable
#' 
#' resample a dataframe to counteract variable imbalance
#' 
#' @param x data frame to extend
#' @param variable column name
#' @param method one of permissible methods listed in function (will print if wrong method passed)
#' @param minimum_ratio minimum acceptable ratio of smallest to largest group n
#' 
#' @return new data frame
#' @author Avants BB
#' @export
balanceDataMultiClass <- function( x, variable, method, minimum_ratio=0.99 ) {
  if ( method == 'none' ) return( x )
  x[,variable]=as.character( x[,variable] )
  if ( method == 'under') {
      minmaxdifftbl = table( x[,variable] )
      mintbl = min( minmaxdifftbl )
      diffis = min( minmaxdifftbl ) / max( minmaxdifftbl )
      while( diffis < minimum_ratio ) {
          minmaxdifftbl = table( x[,variable] )
          maxtbl = max( minmaxdifftbl )
          mintbl = min( minmaxdifftbl )
          diffis = mintbl/maxtbl
          if ( diffis < minimum_ratio ) {
              maxname = names( minmaxdifftbl[ minmaxdifftbl == max(minmaxdifftbl) ] )
              samplethismany = round( mintbl + ( maxtbl - mintbl ) * minimum_ratio ) - 1
              indsformax = sample( which( x[,variable] == maxname ), samplethismany, replace=FALSE )
              otherinds = which( x[,variable] != maxname )
              x=x[c(otherinds,indsformax),]
              }
          }
      } else if ( method == 'over') {
          minmaxdifftbl = table( x[,variable] )
          mintbl = min( minmaxdifftbl )
          diffis = min( minmaxdifftbl )/max( minmaxdifftbl )
          while( diffis < minimum_ratio ) {
              minmaxdifftbl = table( x[,variable] )
              maxtbl = max( minmaxdifftbl )
              diffis = min( minmaxdifftbl )/max( minmaxdifftbl )
              if ( diffis < minimum_ratio ) {
                  minname = names( minmaxdifftbl[ minmaxdifftbl == min(minmaxdifftbl) ] )
                  samplethismany = round( maxtbl * minimum_ratio )+1
                  indsformax = sample( which( x[,variable] == minname ), samplethismany, replace=TRUE )
                  otherinds = which( x[,variable] != minname )
                  x=x[c(otherinds,indsformax),]
                  }
              }
      } else if ( method == 'mwmote' ) {
        minmaxdifftbl = table( x[,variable] )
        diffis = min( minmaxdifftbl )/max( minmaxdifftbl )
        while ( diffis < minimum_ratio ) {    
            minmaxdifftbl = table( x[,variable] )
            diffis = min( minmaxdifftbl )/max( minmaxdifftbl )
            if ( diffis < minimum_ratio ) {
                samplethismany = round( (max( minmaxdifftbl ) - min( minmaxdifftbl )) * minimum_ratio ) + 1
                temp=imbalance::mwmote( x, samplethismany, classAttr = variable )
                x=rbind( x, temp)
            }
        }
    }
  return( x )
}

#' balanced data partition
#' 
#' caret-based data train-test data partition function compatible with mlr3
#' 
#' @param x target vector to split to train test
#' @param perc percentage ( 0 to 1 ) to place in training partition
#' @param subjectIDs unique IDs per subject; aids with repeated measurement data
#' by ensuring subjects exist in uniquely in either train or test
#' 
#' @return vector of strings
#' @author Avants BB
#' @export
dataPartition <- function( x, perc, subjectIDs=NULL ) {
  if ( is.null( subjectIDs ) ) {
    train=caret::createDataPartition( x, p=perc )$Resample1
    test = c(1:length(x))
    test = test[ ! ( test %in% train) ]
    return( list( train=train, test=test ) )
  } else {
    if ( ! is.numeric( x ) ) {
      xuse = as.numeric( as.factor(x) )
    } else xuse = x
    xdf=data.frame( x=xuse, sid=subjectIDs )
    xdf=aggregate( x ~ sid, data=xdf, median )
    train=caret::createDataPartition( xdf$x, p=perc )$Resample1
    trainIDs = xdf$sid[train]
    testIDs = xdf$sid[-train]
    test = which( subjectIDs %in% testIDs )
    train = which( subjectIDs %in% trainIDs )
    return( list( train=train, test=test ) )
  }
}


#' mlr3classifiers
#' 
#' good classification learners from mlr3 (in my experience)
#' 
#' @param twoclass boolean
#' @param all boolean
#' 
#' @return vector of strings
#' @author Avants BB
#' @export
mlr3classifiers <- function( twoclass=TRUE, all=FALSE ) {
    if ( all ) {
      mymsg = 'Use \n mlr3extralearners::list_mlr3learners(select = c("id", "required_packages")) '
      message(mymsg)
      # myl = mlr3extralearners::list_mlr3learners(select = c("id", "required_packages"))
      # myl = myl[ -grep("RWeka",myl$required_packages), ]
      # myl2 = as.data.table(mlr_learners)
      # myl2 = myl2[ -grep("RWeka",myl2$packages), ]
      return( mymsg )
    }
    mylearners =  paste0( "classif.", 
        c("gbm","glmnet","kknn", 'ranger', 'ksvm',# 'mob',
        # "lssvm", 
        "rpart", "xgboost", #"e1071",
         "randomForest", "fnn",
        # 'liblinear', 
        'naive_bayes' ) )
    twoclassers=c('classif.imbalanced_rfsrc')
    if ( twoclass ) mylearners = c( mylearners, twoclassers )
    return( mylearners )
}



#' mlr3classifier
#' 
#' build a mlr3 classification model
#'
#' @param dfin dataframe input
#' @param tcols columns for the prediction task - first is the target outcome
#' @param learnerName which mlr3 learner to instantiate
#' @param partrate partition ratio for the training 0.8 equals 80 percent train 20 test
#' @param dup_size integer for over/under/smote sampling
#' @param balancing string over, under, smote, rwo, mwmote, racog, none are the options
#' @param subjectIDs unique IDs per subject; aids with repeated measurement data
#' by ensuring subjects exist in uniquely in either train or test
#' @param verbose boolean
#' @return dataframe with task, learner, accuracy and balanced accuracy
#' @author Avants BB
#' @export
mlr3classification <- function( dfin, tcols, learnerName, partrate=0.80, dup_size=0, balancing="smote", subjectIDs=NULL, verbose=TRUE ) {
    tarzan = tcols[1]
    valbal = c("none","over","under","smote", "rwo", "racog", "mwmote" )
    if ( ! ( balancing %in% valbal ) )
        stop(paste("balancing must be one of ", paste(valbal, collapse=" / ")))
    if ( sum( table( dfin[,tarzan] ) > 0 ) == 2 ) {
        twoclass=TRUE 
        dfin[,tarzan]=as.factor(as.character(dfin[,tarzan]))
        } else twoclass=FALSE
    selectviz = subtyper::fs( !is.na( dfin[,tarzan])  )
    trainppmisplit = dfin[ selectviz, tcols ]
    split = dataPartition(trainppmisplit[,tarzan], partrate, subjectIDs )
    task_penguins = as_task_classif( formula(paste(tarzan, " ~ .")), data = trainppmisplit)

    if ( balancing %in% c( "rwo", "racog", "mwmote" ) & dup_size > 1 ) {
      temptbl = table( trainppmisplit[,tarzan] )
      myinstsize = min( temptbl ) * ( dup_size - 1 )
      if ( balancing == 'mwmote')
        balDF = mwmote( dataset = trainppmisplit, numInstances = myinstsize, classAttr = tarzan)
      if ( balancing == 'rwo')
        balDF = rwo( dataset = trainppmisplit, numInstances = myinstsize, classAttr = tarzan)
      if ( balancing == 'racog')
        balDF = racog( dataset = trainppmisplit, numInstances = myinstsize, classAttr = tarzan)
      newrownum=(nrow(trainppmisplit)+1)
      trainppmisplit = rbind( trainppmisplit, balDF )
      split$train = c( split$train, newrownum:nrow(trainppmisplit) )
    }

    # SMOTE enriches the minority class with synthetic data
    if ( balancing == 'smote' )
        gr_smote =
            po("colapply", id = "int_to_num",
                applicator = as.numeric, 
                affect_columns = selector_type("integer")) %>>%
            po("smote", dup_size = dup_size) %>>%
            po("colapply", id = "num_to_int",
                applicator = function(x) as.integer(round(x, 0L)), 
                affect_columns = selector_type("numeric"))
            # enrich minority class by factor (dup_size + 1)

    # oversample majority class (relative to majority class)
    if ( balancing == 'over' )
        gr_smote = po("classbalancing",
            id = "oversample", adjust = "minor",
            reference = "minor", shuffle = FALSE, ratio = dup_size)
            # enrich minority class by factor 'ratio'

    if ( balancing == 'under' )
        gr_smote = po("classbalancing",
            id = "undersample", adjust = "major",
            reference = "major", shuffle = FALSE, ratio = 1 / dup_size )

    jj = learnerName
    learner = lrn( jj )
    if ( dup_size > 0 & jj != "classif.imbalanced_rfsrc" & 
      balancing %in% c('smote','over','under') ) {
        learner = as_learner(gr_smote %>>% learner)
      } 
    learner$train(task_penguins, split$train)
    prediction = learner$predict( task_penguins, split$test )
    measure = msr("classif.bacc")
    bacc=prediction$score(measure)
    measure = msr("classif.acc")
    acc=prediction$score(measure)
       
    return( list( task=task_penguins, split=split, learner=learner, acc=acc, bacc=bacc ) )
}


#' mlr3classifiercv
#' 
#' cross-validate a mlr3 classification model
#'
#' @param dfin dataframe input
#' @param tcols columns for the prediction task - first is the target outcome
#' @param nrepeats number of subsampling-driven train test runs
#' @param partrate partition ratio for the training 0.8 equals 80 percent train 20 test
#' @param dup_size integer for over/under/smote sampling
#' @param balancing string over, under, smote, none are the options
#' @param mylearners the mlr3 learners over which to search ; defaults to those that support multiclass
#' @param subjectIDs unique IDs per subject; aids with repeated measurement data
#' by ensuring subjects exist in uniquely in either train or test
#' @param verbose boolean
#' @return dataframe quantifying performance
#' @author Avants BB
#' @export
mlr3classifiercv <- function( dfin, tcols, nrepeats=10, partrate=0.80, dup_size=0, balancing="smote", mylearners = mlr3classifiers(), subjectIDs=NULL, verbose=TRUE ) {
    tarzan = tcols[1]
    valbal = c("none","over","under","smote", "rwo", "racog", "mwmote" )
    if ( ! ( balancing %in% valbal ) )
        stop(paste("balancing must be one of ", paste(valbal, collapse=" / ")))
    if ( sum( table( dfin[,tarzan] ) > 0 ) == 2 ) {
        twoclass=TRUE 
        dfin[,tarzan]=as.factor(as.character(dfin[,tarzan]))
        } else twoclass=FALSE
    selectviz = subtyper::fs( !is.na( dfin[,tarzan])  )
    trainppmi = dfin[ selectviz, tcols ]
    print( table( trainppmi[,tarzan] ))
    clsdf = data.frame()
    for ( r in 1:nrepeats ) {
        if ( verbose ) print(paste("repeat",r))
        split = dataPartition(trainppmi[,tarzan], partrate, subjectIDs )
        trainppmisplit = trainppmi
        if ( balancing %in% c( "rwo", "racog", "mwmote" )  ) {
          temptbl = table( trainppmisplit[,tarzan] )
          myinstsize = max( temptbl ) - min( temptbl )
          if ( balancing == 'mwmote')
            balDF = mwmote( dataset = trainppmisplit[split$train,], numInstances = myinstsize, classAttr = tarzan)
          if ( balancing == 'rwo')
            balDF = rwo( dataset = trainppmisplit[split$train,], numInstances = myinstsize, classAttr = tarzan)
          if ( balancing == 'racog')
            balDF = racog( dataset = trainppmisplit[split$train,], numInstances = myinstsize, classAttr = tarzan)
          newrownum=(nrow(trainppmisplit)+1)
          print( paste("adding",nrow(balDF),"extrapolated data via", balancing) )
          trainppmisplit = rbind( trainppmisplit, balDF )
          split$train = c( split$train, newrownum:nrow(trainppmisplit) )
        }

        task_penguins = as_task_classif( formula(paste(tarzan, " ~ .")), data = trainppmisplit)

        # SMOTE enriches the minority class with synthetic data
        if ( balancing == 'smote' )
        gr_smote =
            po("colapply", id = "int_to_num",
                applicator = as.numeric, 
                affect_columns = selector_type("integer")) %>>%
            po("smote", dup_size = dup_size) %>>%
            po("colapply", id = "num_to_int",
                applicator = function(x) as.integer(round(x, 0L)), 
                affect_columns = selector_type("numeric"))
            # enrich minority class by factor (dup_size + 1)

        # oversample majority class (relative to majority class)
        if ( balancing == 'over' )
        gr_smote = po("classbalancing",
            id = "oversample", adjust = "minor",
            reference = "minor", shuffle = FALSE, ratio = dup_size)
            # enrich minority class by factor 'ratio'

        if ( balancing == 'under' )
        gr_smote = po("classbalancing",
            id = "undersample", adjust = "major",
            reference = "major", shuffle = FALSE, ratio = 1 / dup_size )

        for ( jj in mylearners ) {
            if ( verbose ) print(paste("mylearners",jj))
            learner = lrn( jj )
            if ( dup_size > 0 & jj != "classif.imbalanced_rfsrc" & 
                balancing %in% c('smote','over','under') )
                learner = as_learner(gr_smote %>>% learner)
            learner$train(task_penguins, split$train)
            prediction = learner$predict( task_penguins, split$test )
            if ( verbose )
              print( prediction$confusion )
            measure = msr("classif.bacc")
            bacc=prediction$score(measure)
            measure = msr("classif.acc")
            acc=prediction$score(measure)
            n=nrow(clsdf)+1
            clsdf[n,'mdl']=jj
            clsdf[n,'bacc']=bacc
            clsdf[n,'acc']=acc
            clsdf[n,'r']=r
            }
        }
       
    return( clsdf )
}



#' fill1col2another
#' 
#' fill missing data values into one column from another column;
#' will only fill values that are NA
#'
#' @param df dataframe input
#' @param x column to fill
#' @param y column to fill from
#' @return filled dataframe
#' @author Avants BB
#' @export
fill1col2another <- function( df, x, y ) {
    losel = is.na( df[,x] ) & !is.na(  df[,y] )
    losel[ is.na( losel )] = FALSE
    if ( sum( losel ) > 0 )
      df[ losel, x ] = df[ losel, y ]
    return( df )
    }



#' dcurvarsel
#' 
#' variable selection with CUR - using default parameters from package example
#'
#' @param curdf dataframe input
#' @param variablenames columns for the task
#' @param fraction scalar between zero and one
#' @return vector of selected variables
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' vtosel = getNamesFromDataframe("Random",mydf)
#' # vimp = dcurvarsel( mydf, vtosel, 0.5 )
#' @export
dcurvarsel <- function( curdf, variablenames, fraction ) {
  curEnv=environment()
  myk = min(c(20, dim(curdf[,variablenames])))-1
  loccur = dCUR::CUR
  environment(loccur) <- curEnv
  pdf(file="/dev/null")
  result = loccur( 
              data=curdf, 
              variables=variablenames,
              k=myk, rows = 1, columns = .2, standardize = TRUE,
              cur_method = "mixture" )
  temp=capture.output(dev.off())
  return( head( result$leverage_columns_sorted$var_names, 
    round(length(variablenames)*fraction) ) )
  }


#' clearcolname
#' 
#' remove a column from a dataframe
#'
#' @param mydf dataframe input
#' @param mycolname columns for the task
#' @return trimmed dataframe
#' @author Avants BB
#' @export
clearcolname = function( mydf, mycolname ) {
    if ( mycolname %in% colnames( mydf ) ) {
        mydf=mydf[,-grep(mycolname,colnames( mydf))]
    }
    return( mydf )
}



#' consensusSubtypingTrain
#' 
#' apply several clustering methods and append results to a dataframe;
#' usually will include one subject per row
#'
#' @param dataToClust dataframe input that contains the relevant variables (may have others as well)
#' @param featureNames names to use in the clustering
#' @param clustVec names of the clustering methods to use
#' @param ktrain the number of clusters
#' @param reorderingVariable the name of the column to use to reorder the cluster names
#' @param mvcl character prefix for the new cluster column names
#' @param ksearch the cluster number(s) for clustering of the methods by concordance
#' @param verbose boolean
#' @return a list with newdata: dataframe with new variables attached; models contains the trained models; reorderers contains a dataframe for reordering cluster levels; quality measurements and clustering based on cluster concordance (adjusted rand index) are also returned.
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' @export
consensusSubtypingTrain = function( dataToClust, featureNames, clustVec, ktrain, reorderingVariable, mvcl='MVST', ksearch=3, verbose=FALSE ) {

    # from mclust - avoid import for this 1 function
    adjustedrandindex = function( x, y ) { 
        x <- as.vector(x)
        y <- as.vector(y)
        if (length(x) != length(y)) 
            stop("arguments must be vectors of the same length")
        tab <- table(x, y)
        if (all(dim(tab) == c(1, 1))) 
            return(1)
        a <- sum(choose(tab, 2))
        b <- sum(choose(rowSums(tab), 2)) - a
        c <- sum(choose(colSums(tab), 2)) - a
        d <- choose(sum(tab), 2) - a - b - c
        ARI <- (a - (a + b) * (a + c)/(a + b + c + d))/((a + b + 
            a + c)/2 - (a + b) * (a + c)/(a + b + c + d))
        return(ARI)
      }

    if ( missing( reorderingVariable ) ) reorderingVariable = featureNames[1]
    stopifnot( reorderingVariable %in% colnames(dataToClust) )
    stopifnot( all( featureNames %in% colnames(dataToClust) ) )
    clustmodels = list()
    reoModels = list()
    newclustnames = paste0(mvcl,"_",clustVec)
    successfulclust = rep( FALSE, length(clustVec) )
    names(successfulclust)=clustVec
    for ( myclust in clustVec ) {
        if ( verbose ) print(paste("Train:",mvcl,myclust))
        mvclLocal = paste0(mvcl,"_",myclust)
        clustmodels[[ myclust ]] =
            trainSubtypeClusterMulti( dataToClust, 
                featureNames, myclust, desiredk=ktrain )
        turkeyshoot = predictSubtypeClusterMulti( dataToClust, 
            featureNames, clustmodels[[ myclust ]], mvclLocal )
        if ( length(table(turkeyshoot[,mvclLocal])) == ktrain ) {
            reodf = reorderingDataframe( turkeyshoot, mvclLocal, reorderingVariable )
            reodfCheck = reodf
            ct = 0
            if ( verbose )
              print(paste("st:",mvclLocal))
            while ( ! all( reodfCheck$x == reodfCheck$ord_x ) & ct < 3 ) {
                temp = predictSubtypeClusterMulti( dataToClust, 
                    featureNames, clustmodels[[ myclust ]], mvclLocal,reorderingDataframe=reodf )
                reodfCheck = reorderingDataframe( temp, mvclLocal, reorderingVariable )
                if (  ! all( reodfCheck$x == reodfCheck$ord_x )  ) reodf=reodfCheck
                ct = ct + 1
                }
            reodf$method=myclust
            reoModels[[myclust]]=reodf
            dataToClust = clearcolname(dataToClust, mvclLocal )
            dataToClust = predictSubtypeClusterMulti( dataToClust, 
                    featureNames, clustmodels[[ myclust ]], mvclLocal,reorderingDataframe=reodf )
            successfulclust[myclust]=TRUE
            }
        }


  clustVecGood=names(successfulclust[successfulclust])
  nn = length(clustVec[successfulclust])
  mystat = matrix(nrow=nn,ncol=nn)
  colnames(mystat)=rownames(mystat)=clustVecGood
  for ( j in newclustnames ) {
    if ( j %in% colnames(dataToClust) )
    for ( k in newclustnames ) {
        if ( k != j & k %in% colnames(dataToClust)  ) {
            cl1 = dataToClust[,j]
            cl2 = dataToClust[,k]
            myari = adjustedrandindex(cl1,cl2)
            # print(paste("j",j,"k",k,"ARI",myari))
            mystat[gsub(paste0(mvcl,"_"),"",j),gsub(paste0(mvcl,"_"),"",k)]=myari
#            else if ( agreestat == 'conf') {
#                cm <- caret::confusionMatrix(factor(cl1), reference = factor(cl2))
#                mystat[j,k]=cm$overall[1]
#            } else if ( agreestat == 'kap' ) mystat[j,k]=psych::cohen.kappa(cbind(cl1,cl2))$kappa
#            mystat[j,k]=chisq.test(cl1,cl2,simulate.p.value = TRUE)$statistic
#            print(paste("chi statistic=",chisq.test(cl1,cl2)$statistic ))
            }
          }
        }
    mymeans = apply( mystat, FUN=mean, MARGIN=2, na.rm=T)
    mymaxes = apply( mystat, FUN=max, MARGIN=2, na.rm=T)
    mostdisagreeable = names(mymeans[fs(mymeans==min(mymeans,na.rm=T))])
    mostagreeable = names(mymeans[fs(mymeans==max(mymeans,na.rm=T))])
    if ( verbose ) {
      message(paste("The most agreeable method is:",mostagreeable))
      print(paste("The most agreeable method is:",mostagreeable))
      message(paste("The most disagreeable method is:",mostdisagreeable))
      print(paste("The most disagreeable method is:",mostdisagreeable))
    }
    mystat2=mystat
    diag(mystat2)=NA
    if ( verbose ) pheatmap::pheatmap( mystat2 )
    if ( any( is.na( ksearch ) ) ) ksearch = 2:round(nn/2)
    mypam = fpc::pamk(mystat2,krange=ksearch,criterion="asw", usepam=TRUE,
        scaling=FALSE, alpha=0.001, critout=FALSE, ns=10 )
    myClusterCluster = mypam$pamobject$clustering
    myClusterScores = rowMeans(mypam$pamobject$medoids,na.rm=T)
    clustind = which( myClusterScores == 
        sort(myClusterScores)[length(myClusterScores) ] )# 2nd best
    moreagreeable = names( myClusterCluster[ myClusterCluster == clustind ] )
    if ( verbose ) {
      message("ClusterCluster")
      print("ClusterCluster")
      print(myClusterCluster)
      print("ClusterSimilarityScores")
      print( myClusterScores )
      message("highestARIMethods")
      print("highestARIMethods")
      print(moreagreeable)
      } 

    return( list( 
            newdata = dataToClust,
            models = clustmodels,
            reorderers = reoModels,
            highestARIMethods = moreagreeable,
            ClusterClusters = myClusterCluster,
            ClusterSimilarityScores = myClusterScores
            )
        )

    }

#' consensusSubtypingPredict
#' 
#' apply several clustering methods and append results to a dataframe;
#' assumes trained models already exist.
#'
#' @param dataToClust dataframe input that contains the relevant variables (may have others as well)
#' @param featureNames names to use in the clustering
#' @param clustVec names of the clustering methods to use
#' @param clustmodels the trained models
#' @param reorderers the reordering data frames
#' @param mvcl character prefix for the new cluster column names
#' @param idvar variable name for unique subject identifier column
#' @param visitName the column name defining the visit variables
#' @param baselineVisit the string naming the baseline visit
#' @return new dataframe with new variables attached
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' @export
consensusSubtypingPredict = function( dataToClust, featureNames, clustVec, clustmodels, 
  reorderers, mvcl, idvar, visitName, baselineVisit  ) {
    stopifnot( all(featureNames %in% colnames(dataToClust)))
    namestoclear = getNamesFromDataframe(mvcl,dataToClust)
    for ( nm in namestoclear )
        dataToClust = clearcolname(dataToClust, nm )
    for ( myclust in clustVec ) {
        mvclLocal = paste0(mvcl,"_",myclust)
        if ( myclust %in% names(reorderers) & myclust %in% names(clustmodels) ) {
          if ( ! missing(idvar) & ! missing(visitName) & ! missing( baselineVisit ) ) {
            stopifnot( all(c(idvar,visitName) %in% colnames(dataToClust)))
            stopifnot( any( baselineVisit %in% dataToClust[,visitName] ) )
            dataToClust = predictSubtypeClusterMulti( dataToClust, 
                    featureNames, clustmodels[[ myclust ]], mvclLocal, 
                    idvar, visitName, baselineVisit, 
                    reorderingDataframe=reorderers[[myclust]] )
          } else {
            dataToClust = predictSubtypeClusterMulti( dataToClust, 
                    featureNames, clustmodels[[ myclust ]], mvclLocal, 
                    reorderingDataframe=reorderers[[myclust]] )
          }
        }
      }
    return( dataToClust )
    }


#' consensusSubtypingPrep
#' 
#' generate input for consensus clustering give several clustering algorithms
#'
#' @param dataToTrain dataframe input that contains the relevant variables (may have others as well) on which training will be based
#' @param dataToPredict dataframe input that contains the relevant variables (may have others as well) for which prediction will be done
#' @param featureNames names to use in the clustering
#' @param clustVec names of the clustering methods to use
#' @param maxK the maximum desired number of classes
#' @param reorderingVariable the name of the column to use to reorder the cluster names
#' @param mvcl character prefix for the new cluster column names
#' @param idvar variable name for unique subject identifier column
#' @param visitName the column name defining the visit variables
#' @param baselineVisit the string naming the baseline visit
#' @param whichrank allows user to get 2nd (or 3rd) rank set of methods
#' @param ntoreturnperk number of method results per k to return
#' @param verbose boolean
#' @return new dataframe with new variables attached
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' @importFrom caret dummyVars contr.ltfr
#' @export
consensusSubtypingPrep = function( dataToTrain, dataToPredict, featureNames, clustVec, maxK, 
  reorderingVariable, mvcl, idvar, visitName, baselineVisit, whichrank=0, ntoreturnperk=2, verbose=FALSE ) {
  ############################
  dataToPredictAll = dataToPredict
  for ( desiredk in 2:maxK ) {
    mvclK=paste0(mvcl,"_cstk_",desiredk)
    ## cluster method training
    cclusterobj = consensusSubtypingTrain( 
        dataToTrain, featureNames, clustVec, desiredk, 
        reorderingVariable=reorderingVariable, mvcl=mvclK, verbose=verbose )
    dataToTrain = cclusterobj[["newdata"]]
    clustmodels = cclusterobj[["models"]]
    reoModels = cclusterobj[["reorderers"]]
    clustlist = names( clustmodels )
    myClusterScores = cclusterobj[["ClusterSimilarityScores"]]
    myClusterCluster = cclusterobj[["ClusterClusters"]]
    clustind = which( myClusterScores == 
        sort(myClusterScores)[length(myClusterScores) - whichrank  ])# 2nd best
    moreagreeable = names( myClusterCluster[ myClusterCluster == clustind ] )
    if ( verbose ) {
      message("moreagreeable")
      print(moreagreeable)
      }
    moreagreeable = head( names( myClusterScores[ order(myClusterScores,decreasing=T) ] ),
      ntoreturnperk )
    if ( verbose ) {
      message("mostagreeable")
      print(moreagreeable)
      }
    # first run the clustering on all PPMI data 
    clustdata = consensusSubtypingPredict( dataToPredict, 
        featureNames, moreagreeable, clustmodels, 
        reoModels, mvclK, 
        idvar=idvar, visitName=visitName, baselineVisit=baselineVisit )
    locclustnames = getNamesFromDataframe( mvclK, clustdata, exclusions="_mem" )
    dataToPredictAll[, locclustnames] = clustdata[, locclustnames ]
    }
    return( dataToPredictAll )
  }


#' consensusSubtypingCOCA
#' 
#' apply consensus clustering give several clustering solutions
#'
#' @param dataToClust dataframe input that contains the relevant variables (may have others as well)
#' @param targetk the desired number of classes
#' @param cocanames names of columns to use for the consensus
#' @param newclustername the column name for the consensus clustering
#' @param reorderingVariable the name of the column to use to reorder the cluster names
#' @param idvar variable name for unique subject identifier column
#' @param visitName the column name defining the visit variables
#' @param baselineVisit the string naming the baseline visit
#' @param maxK maximum number of clusters
#' @param consensusmethod either kmeans or hclust
#' @param returnonehot boolean 
#' @param binnmf integer (0 is default - no nmf) 
#' @param verbose boolean
#' @return new dataframe with new variables attached
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' @importFrom caret dummyVars contr.ltfr
#' @export
consensusSubtypingCOCA = function( dataToClust, targetk, cocanames, newclustername, reorderingVariable, idvar, visitName, baselineVisit, maxK, consensusmethod='kmeans', returnonehot=FALSE, binnmf=0, verbose=TRUE ) {
    # assume we already ran consensuscluster
    if ( !missing(idvar) )
      stopifnot( idvar %in% colnames(dataToClust) )
    if ( !missing(visitName) )
      stopifnot( visitName %in% colnames(dataToClust) )
    if ( !missing(reorderingVariable) )
      stopifnot( reorderingVariable %in% colnames(dataToClust) )
    if ( !missing(baselineVisit) )
      stopifnot( baselineVisit %in% dataToClust[,visitName] )
    stopifnot( all( cocanames %in% colnames(dataToClust) ) )
    usebaseline=FALSE
    if ( !missing(idvar) & !missing(visitName) & !missing(baselineVisit) )
      usebaseline=TRUE
    if ( length(cocanames) == 1 ) {
      dataToClust[,newclustername]=dataToClust[,cocanames]
      message("COCA irrelevant - only a single clustering result is available.")
      return( dataToClust )
    }
    dataToClust = clearcolname(dataToClust, newclustername )
    isbl = rep(TRUE,nrow(dataToClust))
    if ( usebaseline )
      isbl = dataToClust[,visitName] == baselineVisit
    cocoform = paste("~", paste( cocanames, collapse="+" ))
    if ( verbose ) {
        for ( x in cocanames ) {
            print(table(dataToClust[isbl,x] ))
        }
    }
    dmy = dummyVars(cocoform, data = dataToClust[,cocanames])
    dmytx = data.frame(predict(dmy, newdata = dataToClust[isbl,cocanames]))
    if ( binnmf > 1 ) {
      dmytx=nmfbin::nmfbin(data.matrix(dmytx), binnmf )$W
      if ( verbose ) print("nmfbin done")
    }
    if ( returnonehot ) return( dmytx )
    if ( ! missing( targetk ) & missing( maxK ) ) {
      cocatx = coca::coca(dmytx, K = targetk, B=1000, maxIterKM=5000, ccClMethod=consensusmethod )
    } else if ( ! missing( maxK ) ) {
      message(paste(consensusmethod,'spearman',maxK))
      cocatx = coca::coca(dmytx, K = NULL, maxK=maxK, 
#        dunn2s=TRUE,
        choiceKmethod='AUC', 
#        widestGap=TRUE, ccDistHC='spearman',
        B=1000, maxIterKM=5000, ccClMethod=consensusmethod )
    } else stop("Must set either maxK or targetk")
    if ( verbose ) message("COCA complete")
#    cocatx = coca::coca(dmytx, maxK = 6, B=5000 )
#    coca = coca::coca( dmytx, maxK = 10, hclustMethod = "average")
    cocatxlab = as.numeric( cocatx$clusterLabels )
    if ( verbose ) {
      print( table( cocatxlab ) )
      }
    dataToClust[,newclustername]=NA
    dataToClust[isbl,newclustername]=cocatxlab
    if ( usebaseline ) {
        temp = fillBaselineColumn( dataToClust,
            newclustername, 
            idvar, visitName, baselineVisit, 
            fast=TRUE, verbose=FALSE )[[1]]
        dataToClust[rownames(temp),newclustername]=temp[,paste0(newclustername,'_BL')]
    }
    if ( !missing(reorderingVariable) ) {
      # now reorder 
      xdf=aggregate( dataToClust[isbl,reorderingVariable], 
        list(dataToClust[isbl,newclustername]),  
        mean, na.rm=TRUE )
      newreo=order(xdf[,2])
      olabels = dataToClust[,newclustername]
      placeholder = olabels
      for ( jj in 1:nrow(xdf) ) placeholder[ olabels == newreo[jj] ] = xdf[jj,1]
      dataToClust[,newclustername] = placeholder
      xdf=aggregate( dataToClust[isbl,reorderingVariable], list(dataToClust[isbl,newclustername]), mean, na.rm=TRUE )
      if ( verbose ) print(xdf)
      }
    dataToClust[,newclustername] = paste0(newclustername, dataToClust[,newclustername] )
    return( dataToClust )
    }


#' consensusSubtypingPAM
#' 
#' apply consensus clustering give several clustering solutions using PAM as the consensus method
#'
#' @param dataToClust dataframe input that contains the relevant variables (may have others as well)
#' @param targetk the desired number of classes
#' @param cocanames names of columns to use for the consensus
#' @param newclustername the column name for the consensus clustering
#' @param reorderingVariable the name of the column to use to reorder the cluster names
#' @param idvar variable name for unique subject identifier column
#' @param visitName the column name defining the visit variables
#' @param baselineVisit the string naming the baseline visit
#' @param maxK maximum number of clusters
#' @param criterion see fpc pamk
#' @param verbose boolean
#' @return new dataframe with new variables attached
#' @author Avants BB
#' @examples
#' mydf = generateSubtyperData( 100 )
#' @export
consensusSubtypingPAM = function( dataToClust, targetk, cocanames, newclustername, reorderingVariable, idvar, visitName, baselineVisit, maxK, criterion='asw', verbose=TRUE ) {
    if ( !missing(idvar) )
      stopifnot( idvar %in% colnames(dataToClust) )
    if ( !missing(visitName) )
      stopifnot( visitName %in% colnames(dataToClust) )
    if ( !missing(reorderingVariable) )
      stopifnot( reorderingVariable %in% colnames(dataToClust) )
    if ( !missing(baselineVisit) )
      stopifnot( baselineVisit %in% dataToClust[,visitName] )
    stopifnot( all( cocanames %in% colnames(dataToClust) ) )
    usebaseline=FALSE
    if ( !missing(idvar) & !missing(visitName) & !missing(baselineVisit) )
      usebaseline=TRUE
    if ( length(cocanames) == 1 ) {
      dataToClust[,newclustername]=dataToClust[,cocanames]
      message("COCA irrelevant - only a single clustering result is available.")
      return( dataToClust )
    }
    dataToClust = clearcolname(dataToClust, newclustername )
    isbl = rep(TRUE,nrow(dataToClust))
    if ( usebaseline )
      isbl = dataToClust[,visitName] == baselineVisit
    cocoform = paste("~", paste( cocanames, collapse="+" ))
    if ( verbose ) {
        for ( x in cocanames ) {
            print(table(dataToClust[isbl,x] ))
        }
    }
    dmy = dummyVars(cocoform, data = dataToClust[,cocanames])
    dmytx = data.frame(predict(dmy, newdata = dataToClust[isbl,cocanames]))
    if ( ! missing( targetk ) & missing( maxK ) ) {
      cocatx = fpc::pamk( dmytx, targetk, usepam=FALSE, criterion=criterion )
    } else if ( ! missing( maxK ) ) {
      cocatx = fpc::pamk( dmytx, 2:maxK, usepam=FALSE, criterion=criterion )
    } else stop("Must set either maxK or targetk")
    if ( verbose ) message("COCA complete")
    cocatxlab = cocatx$pamobject$clustering
    if ( verbose ) {
      print( table( cocatxlab ) )
      }
    dataToClust[,newclustername]=NA
    dataToClust[isbl,newclustername]=cocatxlab
    if ( usebaseline ) {
        temp = fillBaselineColumn( dataToClust,
            newclustername, 
            idvar, visitName, baselineVisit, 
            fast=TRUE, verbose=FALSE )[[1]]
        dataToClust[rownames(temp),newclustername]=temp[,paste0(newclustername,'_BL')]
    }
    if ( !missing(reorderingVariable) ) {
      # now reorder 
      xdf=aggregate( dataToClust[isbl,reorderingVariable], 
        list(dataToClust[isbl,newclustername]),  
        mean, na.rm=TRUE )
      newreo=order(xdf[,2])
      olabels = dataToClust[,newclustername]
      placeholder = olabels
      for ( jj in 1:nrow(xdf) ) placeholder[ olabels == newreo[jj] ] = xdf[jj,1]
      dataToClust[,newclustername] = placeholder
      xdf=aggregate( dataToClust[isbl,reorderingVariable], list(dataToClust[isbl,newclustername]), mean, na.rm=TRUE )
      if ( verbose ) print(xdf)
      }
    dataToClust[,newclustername] = paste0(newclustername, dataToClust[,newclustername] )
    return( dataToClust )
    }


#' Process Clinical/Demographic and imaging Data for an ADNI Study
#'
#' This function merges two data frames based on a common patient ID and the closest date, ensuring that each row in the first data frame (`dfA`) is matched with the row from the second data frame (`dfB`) that has the closest date for the same patient ID. The final merged data frame includes all columns from both `dfA` and `dfB`, excluding the patient ID and date columns from `dfB` to avoid duplication. The function is designed to handle date columns as Date objects and includes a progress bar to indicate the matching process's progress. EXAMDATE is assumed present in dfA and date present in dfB where date is numerically formatted as YYYYMMDD.
#'
#' @param dfA The first data frame to be merged, expected to contain columns for patient ID and date.   may need \code{dfA$subjectID = dfA$PTID}.
#' @param dfB The second data frame to be merged, expected to contain columns for patient ID and date. Rows from `dfB` are matched to `dfA` based on the closest date for each patient ID.
#' @param patientidcol Character string specifying the column name in both data frames that contains the patient ID. Default is 'subjectID'.
#' @param verbose Logical indicating whether to print additional information during processing, such as the number of common IDs found and dimensions of the data frames before and after merging. Default is TRUE.  setting verbose to 2 provides more feedback.
#'
#' @return Returns a merged data frame with the same number of rows as `dfA` and includes all columns from both `dfA` and `dfB`, with `dfB` columns matched based on the closest date for each patient ID. The date columns from `dfB` are excluded to avoid duplication.
#'
#' @examples
#' # Assuming dfA and dfB are already defined and have 'subjectID' and 'date' columns
#' # merged_df <- merge_ADNI_antspymm_by_closest_date(dfA, dfB)
#' 
#' @importFrom utils txtProgressBar setTxtProgressBar
#' @export
#'
#' @note The function assumes that the date columns in the dfB data frame formatted as 'YYYYMMDD' and converts them to Date objects for processing. The progress bar functionality uses base R's txtProgressBar, which is displayed in the console.  dfA's EXAMDATE is of the form YYYY-MM-DD.
#' @importFrom utils txtProgressBar setTxtProgressBar
#' @export
merge_ADNI_antspymm_by_closest_date <- function(dfA, dfB, patientidcol='subjectID', verbose=TRUE) {
  # Safety checks for required columns in dfA and dfB
  if (!all(c(patientidcol, 'EXAMDATE') %in% names(dfA))) {
    stop("dfA is missing one of the required columns: ", patientidcol, " or EXAMDATE ... try setting dfA$subjectID = dfA$PTID ")
  }
  if (!all(c(patientidcol, 'date') %in% names(dfB))) {
    stop("dfB is missing one of the required columns: ", patientidcol, " or date")
  }
  
  # Attempt to convert dfB's date to Date object
  tryCatch({
    dfB[,'EXAMDATE'] <- as.Date(as.character(dfB[,'date']), format="%Y%m%d")
  }, error = function(e) {
    stop("Error in converting 'date' column in dfB to Date format: ", e$message)
  })
  
  # Intersect patient IDs to ensure matching on patientidcol
  isect <- intersect(dfA[, patientidcol], dfB[, patientidcol])
  if (verbose) {
    print(paste(length(isect), "common ids"))
  }
  dfA <- dfA[dfA[, patientidcol] %in% isect, ]
  dfB <- dfB[dfB[, patientidcol] %in% isect, ]
  
  if (verbose) {
    print("Initial dfA dimensions")
    print(dim(dfA))
    print(head(dfA[, 'EXAMDATE'], 3))
    print(head(dfB[, 'EXAMDATE'], 3))
  }
  
  # Initialize progress bar
  pb <- txtProgressBar(min = 0, max = nrow(dfA), style = 3)
  
  matched_rows <- vector("list", nrow(dfA))
  date_diff_years <- numeric(nrow(dfA))
  
  for (i in 1:nrow(dfA)) {
    current_row <- dfA[i, ]
    matching_rows <- dfB[dfB[, patientidcol] == current_row[, patientidcol], ]
    
    if (nrow(matching_rows) > 0) {
      date_diffs <- abs(difftime(matching_rows[,'EXAMDATE'], current_row[,'EXAMDATE'], units = "days"))
      date_diffs <- as.numeric(date_diffs) / 365.25
      closest_row_index <- which.min(date_diffs)
      matched_rows[[i]] <- matching_rows[closest_row_index, ]
      date_diff_years[i] <- min(date_diffs)
    } else {
      matched_rows[[i]] <- setNames(as.list(rep(NA, ncol(dfB))), names(dfB))
      date_diff_years[i] <- NA
    }
    
    setTxtProgressBar(pb, i)
  }
  
  close(pb)
  dfA$date_diff_years <- date_diff_years
  
  matched_dfB <- do.call(rbind, matched_rows)
  merged_df <- cbind(dfA, matched_dfB[, !(names(matched_dfB) %in% c('EXAMDATE', patientidcol))])
  
  if (verbose) {
    print("Final merged_df dimensions")
    print(dim(merged_df))
  }

  return(merged_df)
}


#' Process Clinical/Demographic and imaging Data for a PPMI Study
#'
#' This function merges, cleans, and enriches clinical and demographic data from a neurological study.
#' It checks for required columns, replaces placeholder values, merges datasets, and computes new variables.
#'
#' @param demog DataFrame containing PPMI demographic information with columns like PATNO, BIRTHDT.  An example filename from LONI would be Demographics_06Feb2024.csv
#' @param ppmidemog0 DataFrame containing curated PPMI clinical data including PATNO, EVENT_ID, age_at_visit, age.  An example filename from LONI would be PPMI_Curated_Data_Cut_Public_20230612_rev.csv.
#' @param pymf DataFrame containing imaging and additional clinical data keyed by subjectID, date and filename
#' @param pymversion A character string identifying the ANTsPyMM pipeline variant.
#' @param saa DataFrame (optional) containing supplemental clinical measurements with PATNO, EVENT_ID and CSFSAA.
#' @param verbose boolean
#' @return A processed DataFrame with merged and enriched clinical and demographic information.
#' @export
#' @examples
#' \dontrun{
#'   processed_data <- process_clinical_demographic_data(demog, ppmidemog0, saa, pymf)
#' }
merge_ppmi_imaging_clinical_demographic_data <- function(demog, ppmidemog0, pymf, pymversion, saa, verbose=TRUE ) {
  # Load required libraries
  library(dplyr)
  library(forcats)
  if ( missing( saa ) ) {
    saa = ppmidemog0[,c("PATNO","EVENT_ID","CSFSAA")]
  }

  # Ensure required columns are present
  stopifnot(all(c("PATNO", "BIRTHDT") %in% names(demog)),
            all(c("PATNO", "EVENT_ID", "age_at_visit", "age") %in% names(ppmidemog0)),
            all(c("PATNO", "EVENT_ID") %in% names(saa)),
            all(c("subjectID", "projectID", "filename","date","T1Hier_resnetGrade") %in% names(pymf)))

  demog[demog=='.']=NA
  saa[saa=='.']=NA
  ppmidemog0[ppmidemog0=='.']=NA
  ppmidemog0 = ppmidemog0[ , !(colnames(ppmidemog0) %in% "CSFSAA")]
  ppmi = merge( ppmidemog0, saa, by=c("PATNO","EVENT_ID"), suffixes = c("",".y"), all.x=TRUE )
  ppmi[ppmi=='.']=NA
  ppmi$yearsbl = ppmi$age_at_visit - ppmi$age
  rm( ppmidemog0 )
  idcols = c("subjectID","date","subjectIDdate", "imageID",'filename')
  idcolsdemog = c("PATNO","visit_date","EVENT_ID")
  clin2 = pymf
  clin2$PATNO = pymf$subjectID

  udx = sort(unique(ppmi$subgroup))
  dxmapper = data.frame( ppmisubgroup=udx )
  rownames(dxmapper)=udx
  for ( gdx in c("GBA","LRRK2","PINK1","SNCA","PRKN") )
    dxmapper[ udx[grep(gdx,udx)],c('genetictype')]=gdx

  dxmapper[is.na(dxmapper$genetictype),'genetictype']='Sporadic'
  dxmapper["Healthy Control",'genetictype']='CN'
  dxmapper["GBA",'jdx']='PDGBA'         # 1
  dxmapper["GBA + Hyposmia",'jdx']='ProdromalGBA'         # 2
  dxmapper["GBA + RBD + Hyposmia",'jdx']='ProdromalGBA'   # 3
  dxmapper["Healthy Control",'jdx']='CN' # 4
  dxmapper["Hyposmia",'jdx']='ProdromalSporadic' # 5
  dxmapper["LRRK2",'jdx']='PDLRRK2'     # 6
  dxmapper["LRRK2 + GBA",'jdx']='PDLRRK2'     # 7
  dxmapper["LRRK2 + GBA + Hyposmia",'jdx']='ProdromalLRRK2'     # 8
  dxmapper["LRRK2 + Hyposmia",'jdx']='ProdromalLRRK2'     # 9
  dxmapper["PINK1",'jdx']='PDPINK1' # 10
  dxmapper["PRKN",'jdx']='PDPRKN'   # 11
  dxmapper["RBD",'jdx']='ProdromalSporadic' # 12
  dxmapper["RBD + Hyposmia",'jdx']='ProdromalSporadic' # 13
  dxmapper["SNCA",'jdx']='PDSNCA' # 14
  dxmapper["SNCA + Hyposmia",'jdx']='ProdromalSNCA' # 15
  dxmapper["Sporadic",'jdx']='PDSporadic'   # 16
  ####################################################################
  clin2ppmi=unique( clin2$PATNO[ clin2$projectID == 'PPMI' ] )
  ppmidx = ppmi[ , c("PATNO", 'COHORT', 'CONCOHORT', "subgroup")]
  ppmidx$CONCOHORT[ ppmidx$CONCOHORT == 1  ]='PD'
  ppmidx$CONCOHORT[ ppmidx$CONCOHORT == 2  ]='CN'
  ppmidx$CONCOHORT[ ppmidx$CONCOHORT == 3  ]='SWEDD'
  ppmidx$CONCOHORT[ ppmidx$CONCOHORT == 4  ]='Prodromal'
  ppmidx$COHORT[ ppmidx$COHORT == 1  ]='PD'
  ppmidx$COHORT[ ppmidx$COHORT == 2  ]='CN'
  ppmidx$COHORT[ ppmidx$COHORT == 3  ]='SWEDD'
  ppmidx$COHORT[ ppmidx$COHORT == 4  ]='Prodromal'
  paireddx = data.frame( PATNO=clin2ppmi )
  rownames(  paireddx )=clin2ppmi
  clin2$joinedDX=NA
  for ( uid in clin2ppmi ) {
          uid = as.character( uid )
          dx2=na.omit(unique( ppmidx$subgroup[ subtyper::fs(ppmidx$PATNO == uid )]))
          dx1=na.omit(as.character(unique( clin2$joinedDX[ subtyper::fs(clin2$PATNO == uid )])))
          dx1new=dxmapper[dx2,'jdx']
          seluid = subtyper::fs(ppmidx$PATNO == uid )
          if ( sum(seluid) > 0 & length(dx2) > 0 ) {
            ccdx=na.omit(unique( ppmidx$CONCOHORT[ subtyper::fs(ppmidx$PATNO == uid )]))
            ccdx2=na.omit(unique( ppmidx$COHORT[ subtyper::fs(ppmidx$PATNO == uid )]))
            if ( length( ccdx) == 0 ) ccdx=ccdx2
            stopifnot( length(ccdx)==1)
            mygroup=dxmapper[dx2,'genetictype']
            if ( dx2 == "Healthy Control" ) mygroup=""
            if ( !is.na(ccdx)) newjdx = paste0(ccdx,mygroup)
            if ( is.na(ccdx)) newjdx = paste0(ccdx2,mygroup)
            clin2$joinedDX[ subtyper::fs(clin2$PATNO == uid )]=newjdx
            paireddx[uid,c('ppmidx','ccdx', 'ccdx2', 'group', 'joinedDX')]=c(dx2,ccdx,ccdx2, mygroup, newjdx)
          }  else if ( verbose ) print(paste("missing",uid))
      }
  paireddx$isConsensus=!is.na(paireddx$ccdx)
  # table( paireddx$ppmidx, paireddx$joinedDX  )
  clin2$joinedDX[ clin2$joinedDX == 'CNGBA'] = 'CN'
  ####################################################################

  # message("HERE WE DEAL WITH STATS ASYN")
  clin2$AsynStatus=NA
  clin2$age_BL=NA
  ppmi$CSFSAA[ ppmi$CSFSAA %in% c(2)]=NA
  uids = as.character(unique( ppmi$PATNO ))
  clin2$PATNO=as.character(clin2$PATNO)
  for ( u in uids ) {
    clin2sel=clin2$PATNO == as.character(u)
    if ( sum(clin2sel) > 0 ) {
      clin2[clin2sel,'age_BL']=min(ppmi[ ppmi$PATNO == as.character(u), 'age'],na.rm=T)
      suppas = unique(ppmi[ ppmi$PATNO == as.character(u), 'CSFSAA'])
      if ( 1 %in% suppas ) {
        suppas='Positive'
      } else if ( 3 %in% suppas ) {
        suppas='PosMSA'
      } else if ( 0 %in% suppas ) {
        suppas='Negative'
      } else suppas=NA
      clin2[clin2sel,'AsynStatus']=suppas
    }
  }
#  table( is.na(clin2$age_BL))
  ####################################
  imagingdate=rep( NA, nrow( clin2 ) )
  for ( k in 1:nrow( clin2 ) ) {
    imagingdate[k] = substr( unlist(strsplit( clin2$filename[k], '-' ))[3],0,4)
    }
  clin2$imaging_year = imagingdate
  testid="102529"
  ppmi$PATNO=as.character(ppmi$PATNO)
  demog$PATNO = as.character( demog$PATNO )
  commonids = unique( intersect( ppmi$PATNO, demog$PATNO ) )
  ppmi = merge( ppmi, demog, by='PATNO', all.x=TRUE )
  rmnames=names(ppmi)[ grep("[.]y",names(ppmi))]
  ppmi=ppmi[,!(names(ppmi) %in% rmnames)]
  # names(ppmi)=gsub("[.]x","",names(ppmi))
  ppmi$birthdate=ppmi$BIRTHDT
  # convert birtdate to NRG format
  for ( k in 1:nrow(ppmi) ) {
          if ( ! is.na( ppmi$BIRTHDT[k] ) ) {
              temp=unlist(strsplit(ppmi$BIRTHDT[k], "/"))
              ppmi$birthdate[k]=as.numeric(paste0(temp[2],temp[1],15)) # add the 15th as the estimated date
          }
      }
  demog$birthdate=NA
  for ( k in 1:nrow(demog) ) {
    if ( ! is.na( demog$BIRTHDT[k] ) ) {
      temp=unlist(strsplit(demog$BIRTHDT[k], "/"))
      demog$birthdate[k]=as.numeric(paste0(temp[2],temp[1],15)) 
      # add the 15th as the estimated date
      }
  }
  # now we can compute age at imaging using imaging date in addition to BIRTHDT
  clin2$ageatimaging=NA
  clin2$EVENT_ID=NA
  clin2$clin_imaging_mismatch=TRUE
  rm(clin2add)
  for ( k in 1:nrow( clin2 ) ) {
      subjectbdate = unique( demog$birthdate[ demog$PATNO == clin2$PATNO[k] ] )
      clin2$ageatimaging[k] = as.numeric( 
              difftime( 
                  nrgDateToRDate( clin2$date[k] ), 
                  nrgDateToRDate( subjectbdate ), units='weeks' )/52.0)
      # find the closest EVENT_ID using ageviz
      agesel=which( ppmi$PATNO == clin2$PATNO[k] )
      if ( length( agesel ) == 0 | is.na( clin2$ageatimaging[k] ) ) {
          next # this is like continue
          }
      locages=ppmi[agesel,'age_at_visit']
      closest=which.min( abs(clin2$ageatimaging[k]-locages))
      wclin = agesel[closest]
      clin2$EVENT_ID[k]=ppmi[wclin,'EVENT_ID.x']
      # select clin
      if ( length(wclin) == 1 ) {
          if ( ! exists("clin2add" ) ) {
              clin2add=names(ppmi)[ !(names(ppmi) %in% names(clin2) )]
              }
          clin2[k,clin2add]=ppmi[wclin,clin2add]
          clin2$clin_imaging_mismatch[k]=FALSE
          } else {
              stop("akred - should not reach this point")
              wclin = which(ppmi$PATNO == clin2$PATNO[k] )
              # find next closest match
              if ( length(wclin) > 0 ) {
                  locclin=ppmi[wclin,]
                  closest=which.min( abs(clin2$ageatimaging[k]-locclin$age_at_visit))
                  clin2[k,clin2add]=locclin[closest,clin2add]
              }
          }
  }
  clin2$yearsbl = NA
  uids = as.character(unique( clin2$PATNO ))
  for ( u in uids ) {
    clin2sel=clin2$PATNO == as.character(u)
    if ( sum(clin2sel) > 0 ) {
      clin2[clin2sel,'age_BL']=min(clin2[clin2sel, 'ageatimaging'],na.rm=T)
    }
  }
  clin2$age_BL[ is.na( clin2$age_BL )] = clin2$age[ is.na( clin2$age_BL )]
  clin2$yearsbl = clin2$ageatimaging - clin2$age_BL
  clin2$brainVolume = rowSums( clin2[,getNamesFromDataframe( c("T1Hier","vol","hemis"), clin2 )])
  clin2$brainVolume = clin2$brainVolume/mean(clin2$brainVolume,na.rm=T)
  clin2$mrimfg[ clin2$mrimfg == ""]="Unk"
  clin2$APOE[ clin2$APOE == "" ] = "e3/e3"
  clin2$abeta = as.numeric( clin2$abeta )
  clin2$tau = as.numeric( clin2$tau )
  clin2$DXSubAsyn = NA
  ###########################################################################
  nna=!is.na( clin2$AsynStatus )
  clin2$DXSubAsyn[nna]=paste0(clin2$joinedDX[nna], clin2$AsynStatus[nna] )
  updrsnames = c( getNamesFromDataframe( 'updrs' , clin2), 
    getNamesFromDataframe( 'moca' , clin2) )
  for ( u in c(updrsnames,"duration_yrs") ) clin2[,u]=as.numeric( clin2[,u])
  clin2bl=clin2[clin2$EVENT_ID=='BL',]
  aggregate( updrs_totscore ~ DXSubAsyn, clin2bl, mean, na.rm=T )
  negcnids = clin2bl$PATNO[ subtyper::fs(clin2bl$DXSubAsyn == 'CNPositive') ]
  if ( verbose ) {
    print( table( 
      clin2bl$subgroup[ subtyper::fs(clin2bl$PATNO %in% negcnids)  ],
      clin2bl$DXSub[ subtyper::fs(clin2bl$PATNO %in% negcnids)  ] ) )
  }

  ##########################
  clin2$commonID=clin2$PATNO
  clin2$commonAge=clin2$age_BL
  clin2$commonSex="Male"
  clin2$commonSex[clin2$SEX ==0 ]="Female"
  clin2$commonEdu=as.numeric( clin2$educ )
  clin2$MOCA=clin2$moca
  clin2$studyName = clin2$projectID
  clin2$hy=as.numeric( clin2$hy )
  clin2b = fillBaselineColumn( clin2,
          c('brainVolume','hy','MOCA',updrsnames), 
          'commonID', 'yearsbl', 0, 
          fast=FALSE, verbose=F )[[1]]

  if ( verbose ) {
    print( table( 
      clin2bl$joinedDX,
      clin2bl$AsynStatus ) )
  }

  return(clin2b)
}




#' Visualize the Joint Effect of Variables in a GLM
#'
#' This function visualizes the joint effect of several variables in a generalized linear model (GLM)
#' by plotting the predicted response over a range of values for the primary predictor variable, while
#' holding other predictors at their median or mode values. It allows specifying a group for focused analysis.
#'
#' @param demogmdl Data frame containing the variables used in the GLM.
#' @param qmdl Fitted GLM object from which predictions will be generated.
#' @param x Character vector specifying the names of the predictor variables, with the first being the primary.
#' @param y The name of the response variable.
#' @param group The name of the group variable or specific group to be analyzed.
#' @param titlestring Title of the plot indicating the focus of the visualization.
#' @param varstoadd optional additional variables to add into the model; otherwise just extract from the equation
#' @param groupvar Optional; the name of the variable in `demogmdl` that defines group membership. Default is 'group'.
#' @param predictorsigns Optional; a named numeric vector indicating the direction of the effect of each predictor.
#' @param jdf_simulation boolean
#' @param xrange explicitly set xrange
#' @param verbose Logical; if TRUE, additional processing information will be printed to the console.
#'
#' @return Generates a plot visualizing the predicted response and confidence intervals across the range of the primary predictor.
#'
#' @examples
#' # Assuming `data` is your dataset, `fit` is a fitted GLM, 
#' # and you're interested in predictors `x1` and `x2`:
#' # visglm(data, fit, c("x1", "x2"), "y", "control", 
#' #   "Visualization for Control Group")
#' #
#' @importFrom ggplot2 ggplot geom_line geom_point
#' @importFrom ciTools add_ci
#' @import ciTools
#' @export
visglm <- function(demogmdl, qmdl, x, y, group=NULL, titlestring='',  varstoadd=NULL, groupvar = 'group', predictorsigns=NULL, jdf_simulation=FALSE, xrange=NULL, verbose = FALSE) {  
  .env <- environment() ## identify the environment of cv.step
  if ( is.null( varstoadd ) ) {
    varstoadd=all.vars( formula( qmdl ) )
  }
  simulateJointDistribution <- function(exampleData, nSimulations, distribution='unf') {
    # Step 1: Compute covariance matrix and perform Cholesky decomposition
    covMatrix <- cov(data.matrix(exampleData))
    cholMatrix <- chol(covMatrix)
    
    # Step 2: Simulate standard normally distributed data
    n <- nrow(exampleData)  # Original number of observations
    p <- ncol(exampleData)  # Original number of variables
    if ( distribution=='normal' ) {
      simDataStdNormal <- matrix(rnorm(nSimulations * p), nrow = nSimulations, ncol = p)
    } else {
      simDataStdNormal <- scale(matrix( 1:(nSimulations * p), nrow = nSimulations, ncol = p),T,T)
    #  simDataStdNormal[ simDataStdNormal > 1.5]=1.5
    #  simDataStdNormal[ simDataStdNormal < -1.5]=-1.5
    }
    # Step 3: Apply the Cholesky matrix to simulated data to preserve correlation
    simDataCorrelated <- simDataStdNormal %*% cholMatrix
    
    # Step 4: Adjust the means of the simulated data
    meanVector <- colMeans(exampleData)  # Mean of each variable in the original data
    simDataAdjusted <- sweep(simDataCorrelated, 2, meanVector, '+')
    
    # Step 5: Convert to a data frame and assign column names
    simDataFrame <- as.data.frame(simDataAdjusted)
    names(simDataFrame) <- names(exampleData)
    
    return(simDataFrame)
  }

  verbfn <- function( x, verbose=FALSE ) {
    if ( verbose ) print(paste("visglm:", x)); 
  }
  verbfn('start',verbose)
  myMode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
    }
  mostfreq <- function( x, na.rm=TRUE ) {
    if ( is.numeric( x ) ) return( mean(x,na.rm=na.rm ) )
    return( myMode( x ) )
  }

  # Check if groupvar exists in data
  if ( !(groupvar %in% colnames(demogmdl) )) {
    errmsg = paste("The group variable", groupvar, "does not exist in the dataset.")
    stop(errmsg)
  }
  verbfn('check group',verbose)
  if (is.null(predictorsigns)) {
    predictorsigns = rep(1,length(x))
    names(predictorsigns)=x
    }
  verbfn('predictorsigns',verbose)
  xrowname=paste(x,collapse='+')
  verbfn('xrowname',verbose)
  tt=x[1]
  n2sim = 500
  timeaxis0 = seq( min(demogmdl[,tt],na.rm=T), max(demogmdl[,tt],na.rm=T),length.out=n2sim)
  if ( predictorsigns[x[1]] < 0 ) timeaxis0=rev(timeaxis0)
  verbfn('confint',verbose)
  myconf = confint(qmdl)
  verbfn('confint done',verbose)
  mycolor='magenta'
  mylev=group
  verbfn('group',verbose)
  mycolor='blue'
  if ( !is.null(group) ) {
    if ( group != 'control' ) mycolor='blue'
    if ( group=='all') {
      psel=rep(TRUE,nrow(demogmdl))
      mylev=group
    } else psel = demogmdl[,groupvar] == group
  } else psel = !is.na( demogmdl[,y] )
  atpreds = list( )
  verbfn('all.vars',verbose)
  if ( missing( varstoadd) ) {
    varstoadd = all.vars(qmdl$formula)[-1]
  }
  verbfn('varstoadd',verbose)
  if ( jdf_simulation ) {
    simmed = simulateJointDistribution( data.matrix(demogmdl[,x]), n2sim )
    colnames(simmed)=x
    }
  if ( is.null( xrange )) {
    loqhiq=range(demogmdl[,x],na.rm=T)
  } else loqhiq = xrange
  for ( zz in  varstoadd ) {
    n = length( atpreds ) + 1
    if ( verbose ) print( paste( n,"of", length(varstoadd),zz))
    if ( zz %in% x ) {
        timeaxis = seq( loqhiq[1], loqhiq[2],length.out=length(timeaxis0))
        if ( predictorsigns[zz] < 0 ) timeaxis=rev(timeaxis)
        atpreds[[ n ]] = timeaxis
        if ( jdf_simulation )
          atpreds[[ n ]] = as.numeric(simmed[,zz])
    } else if ( is.numeric( demogmdl[psel,zz] )) {
        atpreds[[ n ]] = rep( mean(demogmdl[psel,zz],na.rm=T), length(timeaxis0))
    } else {
        mostfreqvar = mostfreq( demogmdl[psel,zz] )
        atpreds[[ n ]] = rep( mostfreqvar, length(timeaxis0))
    }
    names(atpreds)[[n]] = zz
  }
  
  verbfn('x1',verbose)
  verbfn('Ypred',verbose)
  Y <- predict( qmdl, atpreds, type='response', se.fit=TRUE, env=.env )
  
  verbfn('add_ci',verbose)
  demogmdl$imaging_protocol=mostfreq(demogmdl$imaging_protocol)
  demogmdl$commonID=mostfreq(demogmdl$commonID)
  dat1 <- add_ci(demogmdl, qmdl, names = c("lpb", "upb"), alpha = 0.05, nsims = 25, allow.new.levels = TRUE)
  verbfn('Yci',verbose)
  ydelta = Y$fit - Y$se.fit
  Y$ciminus = Y$fit-1.96*Y$se.fit
  Y$ciplus = Y$fit+1.96*Y$se.fit
  verbfn('ciplus',verbose)
  myyvars = c(demogmdl[psel,y],Y$ciminus, Y$ciplus)
  myyvars = demogmdl[,y]
  verbfn('myyvars',verbose)
  rangerx = range(demogmdl[psel,x],na.rm=T)
  scl = c(0.98,1.02)
  rangerx = rangerx * scl
  rangery = range(myyvars,na.rm=T) * scl
  verbfn('ranger',verbose)
  plot(timeaxis0, Y$fit, xlab = xrowname, ylab = y, type='l',
    xlim=loqhiq,
    ylim=rangery, main=paste(group, " ", titlestring))
  verbfn('plotting',verbose)
  lines(timeaxis0, Y$fit, lwd = 2, col = mycolor)
  lines(timeaxis0, Y$ciplus, lwd = 2, col = "red", lty=2)
  lines(timeaxis0, Y$ciminus, lwd = 2, col = "red", lty=2)
  verbfn('plot',verbose)
#  myco=(coefficents(summary(qmdl)))
#  corows=rownames(myco)
#  bestx=which.min( myco[ ])
  if ( length( x ) > 1 ) {
    xpoints = rowMeans( demogmdl[psel,x], na.rm=T )
  } else xpoints = demogmdl[psel,x]
  points( xpoints, demogmdl[psel,y], col=mycolor )
  if ( !is.null( group ) )
  if ( group == 'all' ) {
    notexp=demogmdl[,groupvar]!=group
    points( (demogmdl[notexp,x]), demogmdl[notexp,y], col='magenta' )
  }
}


#' Assign Quality Control Ratings Based on Specific Criteria
#'
#' This function assigns quality control ratings to a dataset based on several
#' criteria related to measurements of a specific anatomical structure, such as the substantia nigra.
#' The ratings are determined by whether the measurements fall within certain thresholds.
#'
#' @param df A data frame containing the dataset to be rated. No default value, must be provided by the user.
#' @param z_coord_col Name of the column for the Z-coordinate measurement. Default is "NM2DMT_NM_substantianigra_z_coordinate".
#' @param volume_col Name of the column for the volume measurement. Default is "NM2DMT_NM_volume_substantianigra".
#' @param avg_col Name of the column for the average measurement. Default is "NM2DMT_NM_avg_substantianigra".
#' @param sd_col Name of the column for the standard deviation measurement. Default is "NM2DMT_NM_sd".
#' @param max_col Name of the column for the maximum value measurement. Default is "NM2DMT_NM_max".
#' @param lohi A numeric vector of length 2 specifying the lower and upper thresholds for the Z-coordinate. Defaults to c(0.3, 0.7).
#' @param volume_threshold A numeric value specifying the threshold for the volume measurement. Default is 500.
#' @param avg_threshold A numeric value specifying the threshold for the average measurement. Default is 2500.
#' @param sd_threshold A numeric value specifying the threshold for the standard deviation measurement. Default is 50.
#' @param max_threshold A numeric value specifying the threshold for the maximum value measurement. Default is 5500.
#' @param verbose  boolean
#' @return The original data frame with additional columns containing the quality control ratings. 1=pass; 0=fail.
#' @examples
#' # Example usage:
#' # df_rated <- assign_qc_ratings(df)
#' # View(df_rated)
#' @export
assign_qc_ratings_NM2DMT <- function(df, 
                              z_coord_col = "NM2DMT_NM_substantianigra_z_coordinate", 
                              volume_col = "NM2DMT_NM_volume_substantianigra", 
                              avg_col = "NM2DMT_NM_avg_substantianigra", 
                              sd_col = "NM2DMT_NM_sd", 
                              max_col = "NM2DMT_NM_max", 
                              lohi = c(0.3, 0.7), 
                              volume_threshold = 500, 
                              avg_threshold = 2500, 
                              sd_threshold = 50, 
                              max_threshold = 5500, verbose=FALSE ) {  
  df$NM_QC_Ratings_Z <- NA
  df$NM_QC_Ratings_Z[subtyper::fs(df[[z_coord_col]] > lohi[1] & df[[z_coord_col]] < lohi[2])] <- 1
  df$NM_QC_Ratings_Z[subtyper::fs(df[[z_coord_col]] <= lohi[1] | df[[z_coord_col]] >= lohi[2])] <- 0
  df$NM_QC_Ratings_SNVol <- NA
  df$NM_QC_Ratings_SNVol[subtyper::fs(df[[volume_col]] < volume_threshold)] <- 0
  df$NM_QC_Ratings_SNVol[subtyper::fs(df[[volume_col]] >= volume_threshold)] <- 1
  df$NM_QC_Ratings_AvgIntensity <- NA
  df$NM_QC_Ratings_AvgIntensity[subtyper::fs(df[[avg_col]] <= avg_threshold)] <- 1
  df$NM_QC_Ratings_AvgIntensity[subtyper::fs(df[[avg_col]] > avg_threshold)] <- 0
  df$NM_QC_Ratings_SDIntensity <- NA
  df$NM_QC_Ratings_SDIntensity[subtyper::fs(df[[sd_col]] > sd_threshold)] <- 0
  df$NM_QC_Ratings_SDIntensity[subtyper::fs(df[[sd_col]] <= sd_threshold)] <- 1
  df$NM_QC_Ratings_MaxIntensity <- NA
  df$NM_QC_Ratings_MaxIntensity[subtyper::fs(df[[max_col]] > max_threshold)] <- 0
  df$NM_QC_Ratings_MaxIntensity[subtyper::fs(df[[max_col]] <= max_threshold)] <- 1
  

  nmqccols=c("NM_QC_Ratings_Z","NM_QC_Ratings_SNVol","NM_QC_Ratings_AvgIntensity","NM_QC_Ratings_SDIntensity","NM_QC_Ratings_MaxIntensity")
  df$NM_QC_Ratings_Failures=length(nmqccols)-rowSums( df[,nmqccols], na.rm=T )
  df$NM_QC_Ratings_Failures[ is.na( df$NM2DMT_NM_max ) ]=NA
  nmqccols=c(nmqccols,'NM_QC_Ratings_Failures')
  # Optionally, return tables of ratings and their proportions
  ratings_table <- table(df$NM_QC_Ratings_Failures)
  if ( verbose ) {
    print( ratings_table )
  }
  return( df[,nmqccols] )
  # list(df = df, ratings_table = ratings_table, ratings_proportion = ratings_proportion)
}


#' Eliminate Non-Unique Columns from a Matrix
#'
#' This function takes a matrix as input and returns a new matrix with all non-unique columns removed.
#' It first transposes the matrix to work with rows instead of columns, making it easier to apply the
#' `duplicated` function. Then, it finds and removes duplicated rows in the transposed matrix, which correspond
#' to non-unique columns in the original matrix. Finally, it transposes the matrix back to its original
#' orientation, now with only unique columns remaining.
#'
#' @param matrix A numeric or character matrix from which to remove non-unique columns.
#' @return A matrix with non-unique columns removed, maintaining the original order of the remaining columns.
#' @examples
#' exampleMatrix <- matrix(c(1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6), nrow = 3, byrow = TRUE)
#' print(exampleMatrix)
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    2    3    1
#' # [2,]    2    3    1    2
#' # [3,]    3    1    2    4
#' 
#' uniqueMatrix <- eliminateNonUniqueColumns(exampleMatrix)
#' print(uniqueMatrix)
#' #      [,1] [,2]
#' # [1,]    1    1
#' # [2,]    2    2
#' # [3,]    3    4
#' @export
eliminateNonUniqueColumns <- function(matrix) {
  # Transpose the matrix to work with rows instead of columns for easier application of the 'duplicated' function
  t_matrix <- t(matrix)
  
  # Find unique rows in the transposed matrix (which correspond to unique columns in the original matrix)
  unique_t_matrix <- t_matrix[!duplicated(t_matrix), ]
  
  # Transpose back to the original orientation, now with non-unique columns removed
  result_matrix <- t(unique_t_matrix)
  
  return(result_matrix)
}



#' Linear Mixed Effects Model Analysis with ANOVA and Effect Size Calculation
#'
#' This function fits two linear mixed effects models: a base model without the main predictor of interest
#' and a full model including the main predictor. It then compares these models using ANOVA to test the
#' significance of adding the main predictor. Additionally, it calculates the effect sizes for the predictor
#' in the full model. This function is designed to facilitate the analysis of data where both fixed and
#' random effects are present, accommodating complex experimental designs.
#' NOTE: this function will scale variables internally to aid coefficient estimate visualization.
#'
#' @param data A data frame containing the variables referenced in the model formulas.
#' @param outcome The name of the dependent variable (outcome) as a string.
#' @param predictor The name of the main predictor variable as a string.
#' @param fixed_effects A string specifying the fixed effects to be included in the model, excluding the main predictor.
#' @param random_effects A string specifying the random effects to be included in the model. we assume that a subject ID is the first entry if this is a vector.
#' @param predictoroperator either a \code{+} or \code{*}
#' @param verbose boolean
#' @return A list containing the fitted full model object, ANOVA model comparison, calculated effect sizes for the
#' predictor, coefficients of the full model, and the count of unique levels in the random effects variable.
#' @examples
#' # Assuming 'data' is your dataset with columns 'outcome', 'predictor', 'fixed_var1', ...,
#' # and 'subject' as the random effect:
#' # results <- lmer_anv_p_and_d(data, "outcome", "predictor", "fixed_var1 + fixed_var2", "subject")
#' # summary(results$full_model) # Full model summary
#' # results$model_comparison # ANOVA comparison
#' # results$effect_sizes # Effect sizes
#' @importFrom lmerTest lmer
#' @importFrom lme4 isSingular 
#' @importFrom stats anova
#' @importFrom stringr str_extract str_match
#' @importFrom effectsize t_to_d
#' @export
lmer_anv_p_and_d <- function(data, outcome, predictor, fixed_effects, random_effects, predictoroperator='*', verbose=FALSE ) {
  # Validate input to ensure variables exist in the data frame
  stopifnot(is.character(outcome), is.character(predictor), is.character(fixed_effects), is.character(random_effects))

  hasConverged <- function (mm) {
#    if ( !inherits(mm, "merMod")) stop("Error: must pass a lmerMod object")
    retval <- NULL    
    if(is.null(unlist(mm@optinfo$conv$lme4))) {
      retval = 1
    }
    else {
      if (isSingular(mm)) {
        retval = 0
      } else {
        retval = -1
      }
    }
    return(retval)
  }
  # Construct the model formulas directly
  base_model_formula <- paste( outcome, "~", convert_to_random_effects(random_effects), "+", fixed_effects )
  if ( verbose ) {
    print("base_model_formula")
    print( base_model_formula )
    }
  full_model_formula <- paste( base_model_formula, predictoroperator, predictor )
  base_model_formula = as.formula( base_model_formula )
  full_model_formula = as.formula( full_model_formula )
  all_vars = all.vars( full_model_formula )
  missing_vars <- setdiff(all_vars, colnames(data))
  if ( verbose ) {
    print( full_model_formula )
    }
  if (length(missing_vars) > 0) {
    stop("The following variables are missing in the data: ", paste(missing_vars, collapse = ", "), ".")
  }
  
  
  # Subset data to exclude rows with NA values for relevant variables
  datasub <- na.omit(data[, all_vars])
  datasub = scale_variables_in_equation( datasub, full_model_formula )

  # Fit the linear mixed models using lmer from the lme4 package
  base_model = tryCatch({
      lmer(base_model_formula, data = datasub, REML = FALSE)
    }, error = function(e) {
      print("ERROR")
      print(e)
      NULL
    })
  if ( is.null(base_model) ) return(NULL)
  full_model = tryCatch({
    lmer(full_model_formula, data = datasub, REML = FALSE)
    }, error = function(e) {
      print("ERROR")
      print(e)
      NULL
    })
  if ( is.null(full_model)  ) return(NULL)
  if ( hasConverged(full_model) != 1 ) return(NULL)
  
  # Perform ANOVA to compare the models
  model_comparison <- anova(base_model, full_model)
  
  # Calculate effect sizes for the full model
  coefs <- summary(full_model)$coefficients
#  ndf <- length(unique(datasub[[random_effects[1]]])) # Now using datasub for N calculation
#  effect_sizes <- effectsize::t_to_d(coefs[, "t value"], rep(ndf, nrow(coefs)))
  ndf <- round( coefs[, "df"] )
  effect_sizes <- effectsize::t_to_d(coefs[, "t value"], ndf )
  effect_sizes <- data.frame(effect_sizes)
  rownames(effect_sizes) <- rownames(coefs)
  searchpred = all.vars(formula( paste0("z~",predictor)))[-1]
  effect_sizes <- effect_sizes[multigrep(searchpred, rownames(effect_sizes)), ]

  # Prepare and return the results
  results <- list(
    full_model = full_model,
    model_comparison = model_comparison,
    effect_sizes = effect_sizes,
    coefficients = coefs,
    n = length(unique(datasub[[random_effects[1]]]))
  )
  
  return(results)
}


#' Forcefully Unload a Package in R
#'
#' This function forcefully unloads a package from R by detaching it from the search path,
#' unloading its namespace, and trying to remove it from all environments.
#'
#' @param pkg The name of the package to unload as a string.
#' @return Logical TRUE if the package was successfully unloaded, FALSE otherwise.
#' @examples
#' \dontrun{
#'   force_unload_package("ggplot2")
#' }
#' @export
force_unload_package <- function(pkg) {
  # Check if the package is loaded
  if (!pkg %in% loadedNamespaces()) {
    message(paste("Package", pkg, "is not loaded."))
    return(FALSE)
  }
  
  # Detach the package from the search path
  search_pkg <- paste("package", pkg, sep = ":")
  while (search_pkg %in% search()) {
    detach(search_pkg, unload = TRUE, character.only = TRUE)
  }
  
  # Try to unload the namespace
  if (pkg %in% loadedNamespaces()) {
    tryCatch({
      unloadNamespace(pkg)
      message(paste("Package", pkg, "unloaded successfully."))
      return(TRUE)
    }, error = function(e) {
      message(paste("Failed to unload package:", pkg))
      return(FALSE)
    })
  } else {
    message(paste("Package", pkg, "was detached but the namespace was not loaded."))
    return(TRUE)
  }
}


#' Linear Regression Analysis with ANOVA and Effect Size Calculation
#'
#' This function fits two linear regression models: a base model without the main predictor of interest
#' and a full model including the main predictor. It then compares these models using ANOVA to test the
#' significance of adding the main predictor. Additionally, it calculates the effect sizes for the predictor
#' in the full model. This function is designed to facilitate the analysis of data.
#' NOTE: this function will scale variables internally to aid coefficient estimate visualization.
#'
#' @param data A data frame containing the variables referenced in the model formulas.
#' @param outcome The name of the dependent variable (outcome) as a string.
#' @param predictor The name of the main predictor variable as a string.
#' @param fixed_effects A string specifying the fixed effects to be included in the model, excluding the main predictor.
#' @param predictoroperator either a \code{+} or \code{*}
#' @param verbose boolean
#' @return A list containing the fitted full model object, ANOVA model comparison, calculated effect sizes for the
#' predictor, coefficients of the full model, and the sample size (n).
#' @examples
#' # Assuming 'data' is your dataset with columns 'outcome', 'predictor', and 'fixed_var1', ...
#' # results <- lm_anv_p_and_d(data, "outcome", "predictor", "fixed_var1 + fixed_var2")
#' # summary(results$full_model) # Full model summary
#' # results$model_comparison # ANOVA comparison
#' # results$effect_sizes # Effect sizes
#' @importFrom stats lm anova
#' @importFrom effectsize t_to_d
#' @export
lm_anv_p_and_d <- function(data, outcome, predictor, fixed_effects, predictoroperator='*', verbose=FALSE) {
  # Validate input to ensure variables exist in the data frame
  stopifnot(is.character(outcome), is.character(predictor), is.character(fixed_effects))
  
  # Construct the model formulas directly
  base_model_formula <- paste(outcome, "~", fixed_effects)
  if (verbose) {
    print("base_model_formula")
    print(base_model_formula)
  }
  full_model_formula <- paste(base_model_formula, predictoroperator, predictor)
  base_model_formula <- as.formula(base_model_formula)
  full_model_formula <- as.formula(full_model_formula)
  all_vars <- all.vars(full_model_formula)
  missing_vars <- setdiff(all_vars, colnames(data))
  if (verbose) {
    print(full_model_formula)
  }
  if (length(missing_vars) > 0) {
    stop("The following variables are missing in the data: ", paste(missing_vars, collapse = ", "), ".")
  }
  
  # Subset data to exclude rows with NA values for relevant variables
  datasub <- na.omit(data[, all_vars])
  
  # Fit the linear models using lm
  base_model <- tryCatch({
    lm(base_model_formula, data = datasub)
  }, error = function(e) {
    print("ERROR")
    print(e)
    NULL
  })
  if (is.null(base_model)) return(NULL)
  
  full_model <- tryCatch({
    lm(full_model_formula, data = datasub)
  }, error = function(e) {
    print("ERROR")
    print(e)
    NULL
  })
  if (is.null(full_model)) return(NULL)
  
  # Perform ANOVA to compare the models
  model_comparison <- anova(base_model, full_model)
  
  # Calculate effect sizes for the full model
  coefs <- summary(full_model)$coefficients
  ndf <- nrow(datasub)
  effect_sizes <- effectsize::t_to_d(coefs[, "t value"], rep(ndf, nrow(coefs)))
  effect_sizes <- data.frame(effect_sizes)
  rownames(effect_sizes) <- rownames(coefs)
  searchpred <- all.vars(formula(paste0("z~", predictor)))[-1]
  effect_sizes <- effect_sizes[multigrep(searchpred, rownames(effect_sizes)), ]
  
  # Prepare and return the results
  results <- list(
    full_model = full_model,
    model_comparison = model_comparison,
    effect_sizes = effect_sizes,
    coefficients = coefs,
    n = ndf,
    base_model=base_model
  )
  
  return(results)
}


#' Set Seed Based on Current Time
#'
#' This function sets the random number generator seed based on the current time,
#' with a fine resolution of seconds. This ensures a different seed is used each time
#' the function is called, provided calls are at least one second apart.
#'
#' @return The numeric value used as the seed, derived from the current time in seconds.
#' @examples
#' seedValue <- setSeedBasedOnTime()
#' print(seedValue)
#' @export
setSeedBasedOnTime <- function() {
  op <- options(digits.secs = 8)
  # Get the current time
  currentTime <- Sys.time()
  
  # Convert the current time to a numeric value
  # numericTime <- as.numeric(currentTime, units = "secs")
  numericTime = as.integer(substr(as.character(Sys.time()),22,200))
  # Use the numeric time as the seed
  set.seed(numericTime)
  
  # Optionally, return the seed value used
  return(numericTime)
}



#' Plot Categorical Data
#'
#' This function creates a visualization for categorical data using ggplot2. It takes a dataset and a column name,
#' generates a frequency table for the categorical data, and plots the frequencies.
#'
#' @param datatoplot A data frame containing the dataset to be plotted.
#' @param columnName The name of the column in the dataset that contains categorical data.
#'
#' @return A ggplot object representing the frequency of each category in the specified column.
#' @export
#'
#' @examples
#' # plotCategoricalData(myData, "myCategoryColumn")
plotCategoricalData <- function(  datatoplot, columnName ) {
  # Convert the frequency table to a data frame for ggplot

  freqDataFrame <- as.data.frame(datatoplot[[columnName]]$FrequencyTable)
  names(freqDataFrame) <- c("Category", "Frequency")
  
  # Add a column to indicate the subject's category
  freqDataFrame$SubjectCategory <- freqDataFrame$Category == datatoplot[[columnName]]$SubjectCategory
  
  # Plot
  ggplot(freqDataFrame, aes(x = Category, y = Frequency, fill = SubjectCategory)) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    scale_fill_manual(values = c("TRUE" = "red", "FALSE" = "lightblue")) +
    theme_minimal() +
    labs(title = paste("Dist.", columnName, "/ Subject (red)"),
         y = "Frequency",
         x = columnName) +
    coord_flip()  # Flip for horizontal bars
}


#' Convert Character Vector to Random Effects Terms for lme4
#'
#' This function takes a vector of strings representing grouping factors and converts 
#' each string into a random effect term formatted for use in lme4 model formulas. 
#' Specifically, it formats them as random intercepts for the specified grouping factors.
#'
#' @param variables A character vector where each element is the name of a grouping factor
#' for which a random intercept will be included in the model.
#'
#' @return A character vector of random effect terms, each formatted as `(1|group)`, 
#' where `group` is replaced with each string from the input vector.
#'
#' @examples
#' grouping_factors <- c("school", "class")
#' random_effects <- convert_to_random_effects(grouping_factors)
#' print(random_effects)
#'
#' @export
convert_to_random_effects <- function(variables) {
  # Check if the input is indeed a vector of strings
  if (!is.character(variables)) {
    stop("The input should be a character vector.")
  }
  
  # Create the random effect terms
  random_effects <- paste0("(1|", variables, ")")
  
  return( paste(random_effects,collapse="+"))
}


#' Normative Summary
#'
#' This function provides a normative summary for a given subject within a dataset. It can be configured
#' to focus on specific columns and can be zoomed into particular data points.
#'
#' @param data A data frame containing the dataset for analysis.
#' @param subjectRow The row number or identifier for the subject of interest within the dataset.
#' @param columns A vector of column names in the dataset that should be summarized.
#' @param zoom A parameter to specify the focus or zoom level of the summary.  zoom (integer) specifies the number of nearest neighbors to use.
#' @param idcolumn column name for the unique subject ID
#' @param sexcol name of the sex column
#' @param agecol name of the age column
#' @param verbose A logical flag indicating whether to print detailed output.
#'
#' @return A summary object containing normative statistics for the specified columns of the subject.
#' @export
#'
#' @examples
#' # normativeSummary(myData, 1, c("Column1", "Column2"), zoom = 1, verbose = TRUE)
normativeSummary <- function(data, subjectRow, columns, zoom, idcolumn='commonID', sexcol='commonSex', agecol='commonAge', verbose=TRUE) {
  if(!is.data.frame(data)) stop("The 'data' input must be a data frame.")
  if(!all(columns %in% names(data))) stop("All specified columns must exist in the data frame.")
  if(subjectRow > nrow(data) || subjectRow < 1) stop("Subject row is out of bounds.")
  
  if ( ! missing( zoom ) ) {
    dataz=find_closest_subjects( data[subjectRow,], data, k=zoom, sexcol, agecol )
    data = do.call(rbind, dataz)
    subjectRow=1
  }
  succcolor='deepskyblue4'#  'dodgerblue1'
  summaryList <- list()
  histList = list()
  
  for (col in columns) {
    columnData <- data[[col]]
#    if ( subtyper::fs(antspymm_vartype(col) %in% c("T1","T2Flair","DTI")) & 'brainVolume' %in% colnames(data)) {
#      columnData=columnData/data$brainVolume
#      if ( verbose ) {
#        print(paste("normalize",col,'by BV'))
#      }
#    }
    isNumeric <- is.numeric(columnData)
    if (isNumeric) {
      # Process numeric data
      meanVal <- mean(columnData, na.rm = TRUE)
      sdVal <- (stats::sd(columnData, na.rm = TRUE))
      subjectScore <- columnData[subjectRow]
      zScore <- (subjectScore - meanVal) / sdVal
      if ( verbose ) {
        print( paste( col, meanVal, sdVal, subjectScore, zScore ) )
      }
      if ( !is.na( subjectScore) ) {
        tTestResult <- t.test(columnData, alternative = "greater", mu = subjectScore)
      } else tTestResult=list(estimate=NA,p.value=NA,statistic=NA)

      ttl=paste( col,'sub. (blue) vs pop.')
      histdf=data.frame( vid=rep("pop",length(columnData)), value=columnData )
      if ( max( histdf$value,na.rm=T ) < 1 ) {
        scl=100/(range( histdf$value,na.rm=T )[2]-range( histdf$value,na.rm=T  )[1])
        histdf$value=histdf$value * scl
        ttl=paste(ttl,"\n : vals scaled by",insight::format_value(scl))
      }
      histdf[subjectRow,'vid']='subject'

      histList[[col]]=gghistogram(histdf, x = 'value',  
        add.params=list(size=1.25,linetype = "dashed"),
        add = "mean", add_density = FALSE, title=ttl, fill='vid', legend='none')

      summaryList[[col]] <- list(
        Mean = meanVal,
        SD = sdVal,
        SubjectScore = subjectScore,
        ZScore = zScore,
        TTestPValue = tTestResult$p.value
      )
    } else {
      # Process categorical data
      freqTable <- table(columnData)
      subjectCategory <- as.character(columnData[subjectRow])
      
      summaryList[[col]] <- list(
        FrequencyTable = freqTable,
        SubjectCategory = subjectCategory
      )
    }
  }
  
  # Assuming 'summaryList' contains the processed data
  for (col in columns) {
    if (!is.numeric(data[[col]])) {
      # Assume 'summaryList' is available from the earlier processing
      freqTable <- summaryList[[col]]$FrequencyTable
      subjectCategory <- summaryList[[col]]$SubjectCategory
      histList[[col]]=plotCategoricalData( summaryList, col)
    }
  }

  # Print summary

  # Visualization: For simplicity, focusing on numeric columns for z-score plot
  numericColumns <- sapply(summaryList, function(x) "ZScore" %in% names(x))
  if (any(numericColumns)) {
    zScores <- sapply(summaryList[numericColumns], function(x) x$ZScore)
    names(zScores) <- names(summaryList)[numericColumns]
    zScoreDataFrame <- data.frame(Column = names(zScores), ZScore = zScores)
    zScoreDataFrame$Column = (zScoreDataFrame$Column )
    if ( verbose )
      print( zScoreDataFrame )
    histList[[paste0(col,".z")]]=ggplot(zScoreDataFrame, aes(x = Column, y = ZScore, fill = ZScore)) +
      geom_bar(stat = "identity") +
      geom_hline(yintercept = 0, linetype = "dashed") +
      scale_fill_gradient2(low = succcolor, high = "red", mid = "white", midpoint = 0) +
      theme_minimal() +
      labs(title = "Z-Scores vs. pop.",
           y = "Z-Score",
           x = "") +
      coord_flip() + theme(legend.key.width = unit(0.005, "npc"))
  } else {
    cat("No numeric data for z-score plot.\n")
  }

  grid.arrange(grobs=histList,ncol=round(sqrt(length(histList))),top=paste("Normative Results:",data[subjectRow,idcolumn]))
  return(summaryList)
}


#' Find Closest Subjects
#'
#' This function identifies the closest subjects in a target dataset to a reference dataset
#' based on specified criteria such as sex and age.
#'
#' @param reference_df A data frame containing the reference dataset.
#' @param target_df A data frame containing the target dataset to search within.
#' @param k The number of closest subjects to find.
#' @param sex_col_name The name of the column in the datasets that contains subjects' sex.
#' @param age_col_name The name of the column in the datasets that contains subjects' age.
#'
#' @return A data frame with the rows from `target_df` that are closest to `reference_df` based on the criteria.
#' @export
#'
#' @examples
#' # find_closest_subjects(referenceData, targetData, 5, "Gender", "Age")
find_closest_subjects <- function(reference_df, target_df, k, sex_col_name, age_col_name) {
  # Error checking for column names
  if(!(sex_col_name %in% names(reference_df)) | !(age_col_name %in% names(reference_df))) {
    stop("Reference data frame must contain the specified 'sex' and 'age' column names.")
  }
  if(!(sex_col_name %in% names(target_df)) | !(age_col_name %in% names(target_df))) {
    stop("Target data frame must contain the specified 'sex' and 'age' column names.")
  }
  if(!is.numeric(k) | k <= 0 | length(k) != 1) {
    stop("'k' must be a positive integer.")
  }
  
  # Initialize an empty list to store the results
  results <- list()
  
  # Loop through each row in the reference data frame
  for (i in 1:nrow(reference_df)) {
    # Extract the current row's sex and age
    current_sex <- reference_df[[sex_col_name]][i]
    current_age <- reference_df[[age_col_name]][i]
    
    # Filter the target data frame for subjects of the same sex
    same_sex_df <- target_df[target_df[[sex_col_name]] == current_sex,]
    
    # Calculate the absolute difference in age between the reference subject and all target subjects
    same_sex_df$age_diff <- abs(same_sex_df[[age_col_name]] - current_age)
    
    # Sort the data frame by age difference
    sorted_subjects <- same_sex_df[order(same_sex_df$age_diff), ]
    
    # Select the top 'k' closest subjects, handling cases where there are fewer than 'k' subjects
    num_rows_to_select <- min(nrow(sorted_subjects), k)
    closest_subjects <- sorted_subjects[1:num_rows_to_select, ]
    
    # Store the result
    results[[i]] <- closest_subjects
  }
  
  # Return the final result
  return(results)
}




#' Replace Values in a Vector
#'
#' This function replaces specified values within a vector with new values.
#'
#' @param vec The original vector in which values need to be replaced.
#' @param old_values A vector containing the old values to be replaced.
#' @param new_values A vector containing the new values to replace the old ones.
#'
#' @return A vector with the old values replaced by the new values.
#' @export
#'
#' @examples
#' replace_values(c(1, 2, 3, 4), c(2, 4), c(20, 40))
replace_values <- function(vec, old_values, new_values) {
  # Ensure old_values and new_values are vectors of the same length
  if (length(old_values) != length(new_values)) {
    stop("old_values and new_values must have the same length")
  }
  
  # Check for NA in old_values to handle it specifically
  for (i in seq_along(old_values)) {
    if (is.na(old_values[i])) {
      vec[is.na(vec)] <- new_values[i]
    } else {
      vec[vec == old_values[i]] <- new_values[i]
    }
  }
  
  return(vec)
}


#' Match Data Frames Using Greedy K-Nearest Neighbors
#'
#' This function matches rows from a smaller dataframe to rows in a larger dataframe based on 
#' specified numerical variables. The matching is performed using a greedy algorithm that 
#' maximizes uniqueness in the matched subset.
#'
#' @param df1 A dataframe that is the smaller set to match from.
#' @param df2 A dataframe that is the larger set to match to.
#' @param match_vars A character vector of column names to match on. These columns should be 
#' numerical or convertible to numerical.
#' 
#' @return A dataframe containing the rows from \code{df2} that are matched to each row in \code{df1}.
#' The returned dataframe will contain only the columns that are common to both input dataframes.
#' 
#' @examples
#' \dontrun{
#' df1 <- data.frame(age_BL = c(30, 40, 50), commonSex = c("M", "F", "M"), MOCA = c(25, 28, 27))
#' df2 <- data.frame(age_BL = c(31, 39, 51, 60), commonSex = c("M", "F", "F", "M"), MOCA = c(26, 27, 29, 28))
#' matched_df <- match_data_frames(df1, df2, c("age_BL", "commonSex", "MOCA"))
#' }
#'
#' @importFrom FNN get.knnx
#' @export
match_data_frames <- function(df1, df2, match_vars) {
  ocolnames <- intersect(colnames(df1), colnames(df2))
  
  # Convert categorical variables to numeric and handle NA values
  convert_to_numeric <- function(df, match_vars) {
    df_num <- as.data.frame(lapply(df[match_vars], function(var) {
      if (is.factor(var) || is.character(var)) {
        var <- as.numeric(factor(var))
      }
      return(as.numeric(var))
    }))
    return(df_num)
  }
  
  df1_num <- convert_to_numeric(df1, match_vars)
  df2_num <- convert_to_numeric(df2, match_vars)
  
  # Standardize the columns to have mean = 0 and sd = 1
  df1_std <- scale(df1_num)
  df2_std <- scale(df2_num)
  
  # Perform KNN to find all neighbors
  knn_result <- get.knnx(data = df2_std, query = df1_std, k = nrow(df2))
  
  # Initialize an empty vector to store matched indices
  matched_indices <- integer(nrow(df1))
  used_indices <- rep(FALSE, nrow(df2))
  
  # Greedily select the best match for each row in df1
  for (i in 1:nrow(df1)) {
    distances <- knn_result$nn.dist[i, ]
    candidates <- knn_result$nn.index[i, ]
    
    # Find the first candidate that hasn't been used yet
    for (j in 1:length(candidates)) {
      if (!used_indices[candidates[j]]) {
        matched_indices[i] <- candidates[j]
        used_indices[candidates[j]] <- TRUE
        break
      }
    }
  }
  
  # Select the matched rows from df2
  matched_df2 <- df2[matched_indices, ]
  
  return(matched_df2[, ocolnames])
}

#' Apply Sinkhorn method to stabilize a correlation matrix
#'
#' This function applies the Sinkhorn algorithm to stabilize a given correlation matrix.
#' 
#' @param corr_matrix A square correlation matrix to be stabilized.
#' @param epsilon Convergence threshold for Sinkhorn iterations (default: 1e-3).
#' @param max_iter Maximum number of Sinkhorn iterations (default: 100).
#' @return A Sinkhorn-stabilized correlation matrix.
#' @export
#'
#' @examples
#' mycor <- abs(cor( matrix(rnorm(100),nrow=10)))
#' sinkhorn_corr <- sinkhorn_method(mycor)
sinkhorn_method <- function(corr_matrix, epsilon = 1e-3, max_iter = 100) {
  # Ensure correlation matrix is square and symmetric
  if (!is.matrix(corr_matrix) || nrow(corr_matrix) != ncol(corr_matrix)) {
    stop("Input matrix must be a square matrix.")
  }
  nms = colnames(corr_matrix)
  # Initialize u and v vectors
  n <- nrow(corr_matrix)
  u <- rep(1/n, n)
  v <- rep(1/n, n)
  
  # Sinkhorn iteration
  iter <- 0
  while (iter < max_iter) {
    u_new <- 1 / (corr_matrix %*% v + epsilon)
    v_new <- 1 / (t(corr_matrix) %*% u_new + epsilon)
    
    # Check convergence
    if (max(abs(u_new - u)) < epsilon && max(abs(v_new - v)) < epsilon) {
      break
    }
    
    u <- u_new
    v <- v_new
    iter <- iter + 1
  }
  
  # Construct Sinkhorn-stabilized correlation matrix
  temp = ( corr_matrix %*% diag(as.numeric(v)) )
  sinkhorn_corr <- diag(as.numeric(u)) %*% temp
  colnames(sinkhorn_corr)=nms
  return(sinkhorn_corr)
}




#' Select important variables based on stabilized correlations
#'
#' This function selects important variables from a dataset based on stabilized correlations
#' computed using the Sinkhorn method.
#' 
#' @param data A data frame containing the variables.
#' @param cols A character vector specifying which columns (variables) to consider.
#' @param threshold Proportion of variables to select based on importance (default: 0.2).
#' @param epsilon Convergence threshold for Sinkhorn iterations (default: 1e-3).
#' @param max_iter Maximum number of Sinkhorn iterations (default: 100).
#' @param use_glasso set a scalar greater than zero to use graphical LASSO; this parameter relates to sparseness levels
#' @param least boolean take least explanatory (most uncorrelated) variables
#' @param return_raw_importance boolean
#' @return A character vector of selected variable names.
#' @export
#'
#' @examples
#' set.seed(123)
#' data <- data.frame(
#'   x1 = rnorm(100),
#'   x2 = rnorm(100),
#'   x3 = rnorm(100),
#'   x4 = rnorm(100)
#' )
#' selected_vars <- select_important_variables(data, 
#'  c("x1", "x2", "x3", "x4"), threshold = 0.3)
#' print(selected_vars)
select_important_variables <- function(data, cols, threshold = 0.5, epsilon = 1e-3, max_iter = 0, use_glasso=0.1, least=FALSE, return_raw_importance=FALSE ) {
  # Compute the correlation matrix
  mycor <- cor(na.omit(data[, cols]))
  nms = colnames(mycor)
  
  # Compute the inverse of the Sinkhorn-corrected correlation matrix (precision matrix)
  if ( use_glasso > 0 ) {
    glasso_result <- glasso(mycor, rho = use_glasso)
    precision_matrix <- glasso_result$wi  # Precision matrix
  } else precision_matrix <- MASS::ginv(mycor)
  rownames(precision_matrix)=nms
  colnames(precision_matrix)=nms
  # Compute partial correlations from the precision matrix
  partial_cor_matrix <- -cov2cor(precision_matrix)
  diag(partial_cor_matrix) <- 1  # Set the diagonal to 1 for partial correlations
  
  # Ensure we work with absolute values for partial correlations
  abs_partial_cor_matrix <- abs(partial_cor_matrix)

  if ( max_iter > 0 ) { # this is to aid variable selection
    abs_partial_cor_matrix <- sinkhorn_method( abs_partial_cor_matrix, epsilon = epsilon, max_iter = max_iter)
  } 
  # Sum the absolute values of partial correlations for each variable
  diag(abs_partial_cor_matrix)=0
  variable_importance <- apply(abs_partial_cor_matrix, 1, mean)
  names(variable_importance)=colnames( abs_partial_cor_matrix )
  if ( return_raw_importance ) {
    return( sort(variable_importance, decreasing = !least) )
  }
  # Determine the threshold for selection
  num_vars_to_select <- round(length(variable_importance) * threshold)
  
  # Select the most important variables
  important_vars <- names(sort(variable_importance, decreasing = !least)[1:num_vars_to_select])
  
  return(important_vars)
}

#' Create a Table 1 for Population Characteristics
#'
#' This function generates a "Table 1" for summarizing population characteristics
#' in an academic paper. It uses the `gtsummary` package for summarizing and 
#' the `gt` package for table customization, including adding stripes.
#' Additionally, if the output format is LaTeX, it generates the corresponding LaTeX code.
#'
#' @param data A data frame containing the dataset.
#' @param summary_vars A character vector of variable names to include in the summary.
#' @param group_var A character string of the variable name to group by.
#' @param title A character string for the table title. Default is "Table 1: Population Characteristics".
#' @param output_format A character string indicating the desired output format, either "gt" or "latex". Default is "gt".
#' @param landscape boolean
#' @param latex_font_size optional font size changes
#' @return A `gt` table object or LaTeX code depending on the `output_format`.
#' @import dplyr
#' @import gtsummary
#' @import gt
#' @examples
#' data(mtcars)
#' mtcars$cyl <- as.factor(mtcars$cyl)
#' summary_vars <- c("mpg", "hp", "wt")
#' group_var <- "cyl"
#' table1_gt <- create_table1(data = mtcars, 
#'                            summary_vars = summary_vars, 
#'                            group_var = group_var, 
#'                            title = "Table 1: Population Characteristics by Cylinder",
#'                            output_format = "gt")
#' print(table1_gt)
#'
#' table1_latex <- create_table1(data = mtcars, 
#'                                summary_vars = summary_vars, 
#'                                group_var = group_var, 
#'                                title = "Table 1: Population Characteristics by Cylinder",
#'                                output_format = "latex")
#' cat(table1_latex)
#' @export
create_table1 <- function(data, summary_vars, group_var, 
                          title = "Table 1: Population Characteristics", 
                          output_format = "gt",
                          landscape = TRUE,
                          latex_font_size = c("12.0pt" = "8.0pt", "14.4pt" = "10.0pt")) {
  library(gt)
  library(gtsummary)
  
  # Check if group_var is a factor, if not convert it to a factor
  if (!is.factor(data[[group_var]])) {
    data[[group_var]] <- as.factor(data[[group_var]])
  }
  
  # Helper function to check if a column is categorical
  is_categorical <- function(df) {
    sapply(df, function(x) is.factor(x) || is.character(x))
  }
  
  # Initialize variables for continuous and categorical columns
  categorical_vars <- NULL
  continuous_vars <- NULL
  
  # Identify which variables are categorical and which are continuous
  categorical_vars_bool <- is_categorical(data[, summary_vars])
  if (sum(categorical_vars_bool) > 0) {
    categorical_vars <- summary_vars[categorical_vars_bool]
  }
  if (sum(!categorical_vars_bool) > 0) {
    continuous_vars <- summary_vars[!categorical_vars_bool]
  }
  # Create summary table using gtsummary
  table1 <- data %>%
    select(all_of(c(summary_vars, group_var)))
  table1_summary <- tbl_summary(table1, 
      by = all_of(group_var), 
      type = list(where(is.numeric) ~ "continuous"),
      missing = "no" ) %>%
    add_p() %>%
    modify_header(label ~ paste0("**",group_var,"**")) %>%
    bold_labels()
  
  # Convert to gt table for further customization
  gt_table1 <- as_gt(table1_summary)
  
  # Add header and stripes to the gt table
  gt_table1 <- gt_table1 %>%
    tab_header(
      title = title
    )
  
  # If output_format is "latex", return the LaTeX version of the table
  if (output_format == "latex") {
    # Convert the gtsummary table to LaTeX
    latex_table1 <- table1_summary %>% as_gt() %>% as_latex()
    latex_table1 = gsub("_"," ",latex_table1,fixed=TRUE)
    # Adjust font sizes
    for (font_size in names(latex_font_size)) {
      latex_table1 <- gsub(font_size, latex_font_size[[font_size]], latex_table1)
    }
    
    if ( landscape ) {
      # Add landscape environment
      latex_table1 <- paste0("\\begin{landscape}\n", latex_table1, "\n\\end{landscape}")
    }
    return(latex_table1)
  }
  
  # If output_format is "gt", return the gt table
  return(gt_table1)
}


#' Rename Columns in a Data Frame
#'
#' This function renames specified columns in a data frame based on a named vector of new column names.
#' It performs checks to ensure that the new names are valid and that the columns to rename exist in the data frame.
#'
#' @param df A data frame in which columns are to be renamed.
#' @param new_names A named vector where the names correspond to the existing column names and the values are the new column names.
#' 
#' @return The data frame with renamed columns.
#' 
#' @examples
#' # Create a sample data frame
#' df <- data.frame(
#'   old_col1 = c(1, 2, 3),
#'   old_col2 = c(4, 5, 6),
#'   old_col3 = c(7, 8, 9)
#' )
#' 
#' # Define the new column names
#' new_names <- c("old_col1" = "new_col1", "old_col2" = "new_col2")
#' 
#' # Rename the columns
#' renamed_df <- rename_columns(df, new_names)
#' print(renamed_df)
#' 
#' @export
rename_columns <- function(df, new_names) {
  # Check if new_names is a named vector
  if (is.null(names(new_names))) {
    stop("new_names must be a named vector")
  }
  
  # Check if columns to rename exist in the data frame
  if (!all(names(new_names) %in% colnames(df))) {
    stop("Some columns to rename do not exist in the data frame")
  }
  
  # Check if new names are not empty
  if (any(new_names == "")) {
    stop("New names cannot be empty")
  }
  
  # Get the indices of the columns to rename
  idx <- match(names(new_names), colnames(df))
  
  # Rename the columns
  colnames(df)[idx] <- new_names
  
  # Return the modified data frame
  df
}

#' Log Parameters of a Function Call
#'
#' This function logs the parameters and the call time of a specified function to a log file. 
#' If any of the parameters are matrices or vectors longer than 5 elements, their values will not be printed in the log.
#'
#' @param func The function to be called and logged.
#' @param logfile The name of the log file where the function call information will be saved. The actual file will be named `logfile_function_calls.log`.
#' @param ... Additional arguments to be passed to the function `func`.
#'
#' @return The result of the function `func`.
#' @export
#'
#' @examples
#' my_function <- function(a, b, c = 10) {
#'   return(a + b + c)
#' }
#' result <- log_parameters(my_function, "my_log", a = 1, b = 2, c = 3)
#' print(result)
#' 
log_parameters <- function(func, logfile, ...) {
  call <- match.call()
  call_time <- Sys.time()
  verbose <- TRUE
  if (verbose) print(call_time)
  
  # Extract function name and arguments
  func_name <- as.character(call[[2]])
  args <- list(...)
  if (verbose) {
    print(func_name)
    print(paste("logging ... ", length(args)))
  }

  # Prepare log entry with appropriate handling of matrix and long vector arguments
  args_logged <- sapply(names(args), function(arg_name) {
    arg_value <- args[[arg_name]]
    arg_type <- class(arg_value)[1]
    
    if (is.matrix(arg_value) | is.data.frame(arg_value)) {
      paste0(arg_name, " = <matrix>, type = ", arg_type, ", dim = ", paste(dim(arg_value), collapse = "x"))
    } else if (is.vector(arg_value) && length(arg_value) > 5) {
      paste0(arg_name, " = <long vector>, type = ", arg_type, ", length = ", length(arg_value))
    } else {
      paste0(arg_name, " = ", arg_value, ", type = ", arg_type)
    }
  }, USE.NAMES = FALSE)
  
  log_entry <- paste0("[", call_time, "] ", func_name, " called with: ", 
                      paste(args_logged, collapse = ", "), "\n")

  if (verbose) print(log_entry)
  
  # Append log entry to a file
  cat(log_entry, file = paste0(logfile, "_function_calls.log"), append = TRUE)
  
  # Call the original function
  result <- do.call(func, args)
  
  # Return result
  return(result)
}







#' Bipartite Variable Match
#'
#' This function matches two sets of variables using a bipartite graph and returns a data frame of the matched pairs.
#'
#' @param data A data frame containing the required columns for matching.
#' @param cog_col Character. The name of the first variable column. Default is "cog".
#' @param voi_col Character. The name of the second variable column. Default is "voi".
#' @param plot Logical. If TRUE, plots the resulting bipartite graph. Default is FALSE.
#' @param weight_type Character. Defines the edge weight type for the bipartite graph. Options are "d" (distance), "-log(p)" (negative log p-value). Default is "-log(p)".
#' @param edge_width_multiplier Numeric. Multiplier for the edge width in the plot. Default is 11.
#' @param directed Logical. If TRUE, creates a directed graph. Default is FALSE.
#'
#' @return A data frame with matched pairs of variables and their associated data.
#' @import igraph
#' @export
#'
#' @examples
#' # Example usage:
#' df <- data.frame(cog = c("cog1", "cog2"), voi = c(1, 2), 
#'  d = c(0.1, 0.2), p = c(0.05, 0.01), multi = c(1, 2))
#' bipartite_variable_match(df, plot = TRUE)
bipartite_variable_match <- function(data, cog_col = "cog", voi_col = "voi", plot = FALSE, weight_type = "-log(p)", edge_width_multiplier = 11, directed = FALSE) {
  
  # Expected columns
  expected_columns <- c(cog_col, voi_col, "d", "p", "multi")
  
  # Check for missing columns
  missing_columns <- setdiff(expected_columns, names(data))
  if (length(missing_columns) > 0) {
    warning("The following expected columns are missing from the data: ", paste(missing_columns, collapse = ", "))
    return(NULL)
  }
  
  df <- data[, expected_columns]
  g <- graph_from_data_frame(df, directed = directed)
  
  # Set vertex types
  V(g)$type <- names(V(g)) %in% df[[cog_col]]
  
  # Set edge weights based on user-defined weight_type
  if (weight_type == "d") {
    E(g)$weight <- df$d
  } else if (weight_type == "-log(p)") {
    E(g)$weight <- -log(df$p)
  } else {
    stop("Invalid weight_type. Choose either 'd' or '-log(p)'.")
  }
  
  # Compute the maximum bipartite match
  matching <- max_bipartite_match(g, weights = E(g)$weight)
  cogmatching <- na.omit(matching$matching[unique(df[[cog_col]])])
  resultdataframe <- data.frame(
    cog = names(cogmatching), 
    voi = as.integer(cogmatching)
  )
  
  # Add additional data to resultdataframe
  for (k in 1:nrow(resultdataframe)) {
    losel <- df[[cog_col]] == resultdataframe$cog[k] & df[[voi_col]] == resultdataframe$voi[k]
    for (j in c("d", "p", "multi")) {
      resultdataframe[k, j] <- df[losel, j]
    }
  }
  
  # Create a subgraph for plotting
  subg <- graph_from_data_frame(resultdataframe, directed = directed)
  E(subg)$weight <- resultdataframe$d
  
  if (plot) {
    plot(subg, edge.width = E(subg)$weight * edge_width_multiplier)
  }
  
  return(resultdataframe)
}



#' Adjust P-values and Return Subsetted Dataframe
#'
#' This function adjusts p-values using a specified method and alpha level. 
#' It successively tests each method along with each alpha value, and if 
#' the sum is zero, it continues to the next method. Otherwise, it returns 
#' the subsetted dataframe.
#'
#' @param resdf A data frame containing at least a column named \code{p} with p-values.
#' @param methods A character vector specifying the methods for p-value adjustment. Default is \code{c('BY', 'BH', 'none', 'none')}.
#' @param alpha A numeric vector specifying the alpha levels for p-value adjustment. Default is \code{c(0.05, 0.05, 0.05, 0.2)}.
#' @return A subsetted dataframe where the adjusted p-values are less than or equal to the corresponding alpha levels.
#' @details The function checks that the lengths of \code{methods} and \code{alpha} are equivalent and that all alpha values are in the range (0, 1). It iterates over the provided methods and alpha values, adjusting the p-values using the \code{p.adjust} function from the \code{stats} package. If the sum of selected values is greater than zero, it returns the subsetted dataframe.
#' @examples
#' # Example dataset
#' set.seed(123)
#' resdf <- data.frame(
#'   p = runif(100, min = 0, max = 1)
#' )
#'
#' # Example usage of adjust_p_values
#' adjusted_df <- adjust_p_values(resdf, 
#'  methods = c('BY', 'BH', 'none', 'none'), 
#'  alpha = c(0.05, 0.05, 0.05, 0.2))
#' head(adjusted_df)
#' @importFrom stats p.adjust
#' @importFrom ggpubr gghistogram
#' @importFrom glasso glasso
#' @importFrom ggplot2 theme_minimal scale_fill_manual geom_bar coord_flip 
#' @export
adjust_p_values <- function(resdf, methods = c('BY', 'BH', 'none', 'none'), 
                            alpha = c(0.05, 0.05, 0.05, 0.2)) {
  # Check if lengths of methods and alpha are equivalent
  if (length(methods) != length(alpha)) {
    stop("The lengths of 'methods' and 'alpha' must be equivalent.")
  }
  
  # Check if all alpha values are in the valid range (0, 1)
  if (any(alpha <= 0 | alpha >= 1)) {
    stop("All alpha values must be in the range (0, 1).")
  }
  
  # Iterate over the provided methods and alpha values
  for (i in seq_along(methods)) {
    qsel <- p.adjust(resdf$p, method = methods[i]) <= alpha[i]
    if (sum(qsel) > 0) {
      return(resdf[qsel, ])
    }
  }
  
  # If no method produces a non-zero subset, return an empty dataframe
  return(resdf[FALSE, ])
}


#' Identify Best Variable of Interest (VOI)
#'
#' This function identifies the best Variable of Interest (VOI) based on the minimum or maximum value of a specified column. 
#' It groups the data by a specified column and selects rows with the minimum or maximum value in another specified column.
#'
#' @param data A data frame containing the dataset.
#' @param group_var A symbol specifying the column name to group by. Defaults to the first column of the data.
#' @param value_var A symbol specifying the column name to use for selecting the minimum or maximum value. Defaults to the second column of the data.
#' @param select_vars A vector of column names to select in the final output. Defaults to all columns.
#' @param selection A character string specifying whether to select the "min" or "max" value. Defaults to "min".
#' 
#' @return A data frame with the best VOIs based on the specified criteria.
#' 
#' @import dplyr
#' @importFrom rlang enquo
#' @export
#'
#' @examples
#' data <- data.frame(cog = c('A', 'A', 'B', 'B'), 
#'                    voi = c('X1', 'X2', 'Y1', 'Y2'), 
#'                    d = c(0.5, 0.7, 0.3, 0.9), 
#'                    p = c(0.01, 0.05, 0.02, 0.03), 
#'                    multi = c(TRUE, FALSE, TRUE, FALSE))
#' identify_best_voi(data, group_var = cog, value_var = p, 
#'   select_vars = c("cog", "voi", "d", "p", "multi"), selection = "min")
identify_best_voi <- function(data, group_var = names(data)[1], value_var = names(data)[2], select_vars = names(data), selection = c("min", "max")) {
  # Convert input columns to symbols
  group_var <- rlang::enquo(group_var)
  value_var <- rlang::enquo(value_var)
  
  # Match selection argument to ensure it's either "min" or "max"
  selection <- match.arg(selection)
  
  # Group by specified variable and select the rows with the minimum or maximum value of the specified column
  best_vois <- data %>%
    group_by(!!group_var) %>%
    filter(if (selection == "min") !!value_var == min(!!value_var, na.rm = TRUE) else !!value_var == max(!!value_var, na.rm = TRUE)) %>%
    select(all_of(select_vars)) %>%
    ungroup()
  
  return(best_vois)
}


#' Truncate or Remove High Values
#'
#' Truncates or removes high values in a given column of a data frame.
#'
#' @param df The input data frame.
#' @param x The column name to truncate or remove high values from.
#' @param t The threshold value (default is 4).
#' @param removeit Logical indicating whether to remove or truncate high values (default is FALSE).
#'
#' @return The modified data frame.
#' @export
truncatehi <- function(df, x, t = 4, removeit = FALSE) {
  # Sort the column in descending order
  sortvec <- sort(df[, x], decreasing = TRUE)
  
  # Get the threshold value
  t <- sortvec[t]
  selection = na2f(df[, x] > t)
  # Truncate high values if removeit is FALSE
  if (!removeit) {
    df[selection, x] <- t
  }
  # Remove high values if removeit is TRUE
  else {
    df <- df[!selection, ]
  }
  
  # Return the modified data frame
  return(df)
}




#' Shorten Names
#'
#' Shorten a vector of names by removing common substrings, applying custom replacements, 
#' replacing "__" with "_", then "_" with ".", removing duplicates, and truncating to a maximum length.
#'
#' @param names A vector of names to shorten.
#' @param max_length The maximum length of the shortened names. Defaults to 20.
#' @param custom_replacements A list of custom replacements to apply. Defaults to NULL.
#'
#' @return A vector of shortened names.
#'
#' @examples
#' names <- c("npsy_BDI_Total", "npsy_BSI.18_TotalRaw", "npsy_CVLTShortDelayFreeRecall_Raw")
#' shortened_names <- shorten_names(names)
#' print(shortened_names)
#' @export
shorten_names <- function(names, max_length = 20, custom_replacements = NULL) {
  # Default replacements
  default_replacements <- list(
    "score" = "scr",
    "composite" = "comp",
    "recall" = "rcl",
    "total" = "tot",
    "raw" = "",
    "free" = "",
    "delay" = "del",
    'npsy_' = ''
  )
  
  # Use custom replacements if provided, otherwise use default
  replacements <- if (is.null(custom_replacements)) default_replacements else custom_replacements
  
  # Find common substrings
  common_substrings <- names %>%
    strsplit(split = "") %>%
    unlist() %>%
    table() %>%
    sort(decreasing = TRUE) %>%
    names()
  
  # Replace common substrings
  for (i in 1:length(common_substrings)) {
    if (nchar(common_substrings[i]) > 2) {
      names <- gsub(common_substrings[i], "", names)
    }
  }
  
  # Convert to lower case
  names <- tolower(names)
  
  # Apply replacements
  for (i in names(replacements)) {
    names <- gsub(i, replacements[[i]], names, ignore.case = TRUE)
  }
  
  # Replace "__" with "_", then "_" with ".", and remove duplicates
  names <- gsub("__", "_", names)
  names <- gsub("_", ".", names)
  names <- gsub("(.)\\1+", "\\1", names)
  
  # Shorten names to max_length
  names <- substr(names, 1, max_length)
  return(names)
}


#' Create a Radar Chart
#'
#' This function creates a radar chart for the specified data.
#'
#' @param data A data frame containing the data to be plotted.
#' @param group_color A named vector of colors to be used for each group in the radar chart.
#' @param meansd A logical value indicating whether to use mean + 2SD and mean as range (default is FALSE).
#'
#' @return A radar chart plot.
#'
#' @examples
#' # group_color <- c("Group 1" = "red", "Group 2" = "blue", "Group 3" = "green")
#' # create_radar_chart(data, group_color)
#' @export
#' @importFrom fmsb radarchart radarchartcirc
create_radar_chart <- function(data, group_color, meansd = TRUE ) {
  # Ensure all columns are numeric
  data <- data.frame(lapply(data, as.numeric))
  
  # Prepare the data for radar chart
  if (meansd) {
# Calculate mean + SD and mean - SD for each group
    max_min <- data.frame()
    group_color_new=c()
    for (group in unique(names(group_color))) {
      group_data <- data[names(group_color) == group, ]
      mn=colMeans( group_data, na.rm=T )
      sdit=apply( group_data, FUN=sd, MARGIN=2, na.rm=T )
      max_min <- rbind( max_min,mn+sdit*1,mn )
      tt=rep( unique(group_color[group] ),2)
      names(tt)=rep(group,2)
      group_color_new = c( group_color_new, tt )
    }
    colnames(max_min) <- colnames(data)
    radarchartcirc( max_min, axistype = 1,
             pcol = group_color_new,
             pfcol = scales::alpha(group_color, 0.25),
             plwd = 4, plty = 1,
             cglcol = "grey", cglty = 1, axislabcol = "grey", 
             caxislabels = seq(0, 20, 5), cglwd = 0.8, 
             vlcex = 1.5, maxmin=FALSE, palcex=2 )
  
        colortbl0=table(names(group_color_new))
        colortbl1=table(group_color_new)
      legend(x = "topright", legend = names(colortbl0), col = group_color_new[names(colortbl0)], lty = 1, lwd = 4, bty = "n")
      return(NULL)
  } else {
    # Set the range for each variable (minimum and maximum)
    max_min <- apply(data, 2, function(x) c(max(x, na.rm = TRUE), min(x, na.rm = TRUE)))
  }
  max_min <- as.data.frame((max_min))
  
  # Add column names to the max_min data frame (only if max_min has the same number of columns as data)
  if (ncol(max_min) != ncol(data)) {
    stop(paste("The number of columns in max_min (", ncol(max_min), ") does not match the number of columns in data (", ncol(data), ")."))
  }
  colnames(max_min) <- colnames(data)
  
  # Add the range to the data
  data_for_plot <- rbind(max_min, data)
  group_color=c(NA,NA,group_color)
  
  # Plot the radar chart for the specified observation
  radarchartcirc(data_for_plot[c(1, 2, 3:nrow(data_for_plot)), ], axistype = 1,
             pcol = group_color,
             pfcol = scales::alpha(group_color, 0.25),
             plwd = 4, plty = 1,
             cglcol = "grey", cglty = 1, axislabcol = "grey", caxislabels = seq(0, 20, 5), cglwd = 0.8,
             vlcex = 1.5)
  
  colortbl0=table(names(group_color))
  colortbl1=table(group_color)
  legend(x = "topright", legend = names(colortbl0), col = group_color[names(colortbl0)], lty = 1, lwd = 4, bty = "n")
}


#' Replace Multiple Columns in Dataframe Based on a Matching Key
#'
#' This function replaces values in multiple columns of a target dataframe 
#' with values from a source dataframe based on a matching key. The function 
#' takes two dataframes, a key column, and a vector of columns to be updated, 
#' and returns the updated target dataframe.
#'
#' @param target_df A dataframe containing the original values.
#' @param source_df A dataframe containing the new values to replace in the target dataframe.
#' @param key The column name (as a string) that is used as the key for matching rows in both dataframes.
#' @param columns_to_update A character vector of column names in `target_df` whose values will be replaced.
#' 
#' @return A dataframe with the specified columns' values updated based on the matching key.
#' 
#' @export
replace_values_by_key <- function(target_df, source_df, key, columns_to_update) {
  # Ensure key exists in both dataframes
  if (!key %in% names(target_df)) {
    stop(paste("Key column", key, "not found in target_df."))
  }
  if (!key %in% names(source_df)) {
    stop(paste("Key column", key, "not found in source_df."))
  }

  # Find the intersection of columns to update that exist in both dataframes
  valid_columns <- intersect(columns_to_update, intersect(names(target_df), names(source_df)))
  
  # Select only the key and valid columns from the source_df
  source_df_subset <- source_df %>%
    select(all_of(c(key, valid_columns)))
  
  # Join the dataframes
  merged_df <- left_join(target_df, source_df_subset, by = key, suffix = c("", ".new"))
  
  # Iterate over each valid column to update
  for (col in valid_columns) {
    new_col <- paste0(col, ".new")
    merged_df <- merged_df %>%
      mutate(
        !!col := ifelse(!is.na(.data[[new_col]]), 
                        .data[[new_col]], 
                        .data[[col]])
      )
  }
  
  # Remove the temporary new columns
  merged_df <- merged_df %>%
    select(-matches("\\.new$"))
  
  return(merged_df)
}


#' Row-wise linear adjustments to variables in a dataframe
#'
#' Trains linear models on a given dataframe row by row and applies the learned parameters 
#' to perform subject-specific adjustment to other variables.  
#'
#' This function loops through each row in the dataframe, trains a linear model using trainvalues,
#' and applies the model to predict the values for the specified variables. It handles both simple (one predictor)
#' and multiple predictor cases, adapting the modeling approach accordingly and allows for 
#' both linear and polynomial adjustment.
#'
#' @param tempNM The input dataframe containing the data to be processed.
#'
#' @param trainvalues A vector of target values corresponding to the predictor variables. this 
#' should be a vector of values with column names in tempNM representing the predictor variables.
#'
#' @param varstoadjust A vector of column names in tempNM representing the variables to be predicted.
#'
#' @param poly_order An integer specifying the polynomial order to be used in the fit. Defaults to 1.
#' @param post_fix an extension to add to the modified variable name - can be an empty string
#' @param verbose boolean
#'
#' @return A dataframe with the predicted values added as new columns. The new columns are named by appending post_fix
#' to the original column names specified in varstoadjust.
#'
#' @details
#' The function uses a simple linear model for prediction, which may not be suitable for complex relationships between variables.
#' It assumes that the input dataframe has the same structure and column names as expected, and does not perform any error checking
#' or handling. Therefore, it may fail if the input data is not properly formatted or if there are missing values.
#'
#' @examples
#' trainvec = colMeans(mtcars[,1:7])
#' mtadj = rowwise_linear_variable_adjustments(mtcars, trainvec, c("mpg", "cyl","disp") )
#' # plot( mtadj[,'disp_pred'], mtcars[,'disp'] )
#' @export
rowwise_linear_variable_adjustments <- function(tempNM, trainvalues, varstoadjust,
  poly_order=1, post_fix = '_pred', verbose=FALSE ) {

  learn_linear_mapping <- function(ground_truth, variation, new_vector = NULL, poly_order = 2) {
    # Create a polynomial design matrix
    poly_design <- stats::poly(variation, degree = poly_order, raw = FALSE)
    
    # Fit a linear model using the polynomial design matrix and the ground truth
    model <- lm(ground_truth ~ poly_design)
    
    # If a new vector is provided, apply the learned coefficients to transform it
    if (!is.null(new_vector)) {
      new_poly_design <- stats::poly(new_vector, degree = poly_order, raw = FALSE)
      transformed_new_vector2 <- coef(model)[1] + as.matrix(new_poly_design) %*% coef(model)[-1]
      return( as.numeric(transformed_new_vector2 ))
    } else {
      return(model)
    }
  }

  # Create a copy of the dataframe to store predictions
  result_df <- data.frame()
  intvars = names( trainvalues )
  # Loop over each row in the dataframe
  nth = round( nrow(tempNM)/100)
  for (i in 1:nrow(tempNM)) {
    if ( verbose & i %% nth == 0 ) cat(paste0(i,'...'))
    # Extract the current row's predictor values
    current_row <- as.numeric(tempNM[i, intvars])
    current_test <- as.numeric(tempNM[i, varstoadjust])
    # Store the predictions in the result dataframe
    learnedmap = learn_linear_mapping( trainvalues, current_row, current_test, poly_order=poly_order )
    result_df[i, paste0(varstoadjust, post_fix )] = as.numeric(learnedmap)
  }
  cat("\n")
  
  return(result_df)
}







#' Create a new vector with a specified correlation to an input vector
#'
#' @param a Input vector
#' @param rho Desired correlation between a and the new vector
#'
#' @return A new vector with the desired correlation to a
#'
#' @examples
#' a <- rnorm(100)
#' rho <- 0.7
#' b <- create_correlated_vector(a, rho)
#' cor(a, b)
#' @export
create_correlated_vector <- function(a, rho) {
  stopifnot(rho >= -1, rho <= 1)
  e <- rnorm(length(a))
  b <- rho * (a - mean(a)) / sd(a) + sqrt(1 - rho^2) * e
  mean(a) + sd(a) * b / sd(b)
}


#' Generate a table summarizing the data in a data frame.
#'
#' @param df Data frame containing the data.
#' @param vars Vector of variable names to include in the summary.
#' @param facet_var Name of the facet variable.
#' @param col_names Vector of custom column names (optional).
#' @param factor_rows Logical indicating whether to display factor variables in rows (default = FALSE).
#'
#' @return A data frame summarizing the data in the formula.
#'
#' @examples
#' \dontrun{
#' table_1(df, vars, facet_var)
#' }
#' @importFrom stats sd
#' @export
table_1 <- function(df, vars, facet_var, col_names = NULL, factor_rows = FALSE) {
  # Define a function to format the output for numeric variables
  format_output <- function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) {
      return("NA (NA)")
    } else {
      mean_sd <- paste0(round(mean(x), 2), " (", round(sd(x), 2), ")")
      return(mean_sd)
    }
  }
  
  # Define a function to format the output for factors
  format_factor <- function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) {
      return("NA")
    } else {
      freq <- table(x)
      result <- paste0(round(freq / sum(freq) * 100, 2), collapse = "/")
      return(paste0(result, "%"))
    }
  }
  
  # Split the data by the facet variable
  df_split <- split(df, df[, facet_var])
  
  # Initialize results
  row_names <- c("Sample.Size")
  results <- list()
  
  # Add sample size row
  sample_sizes <- sapply(df_split, function(x) sum(!is.na(x[, facet_var])))
  results[[1]] <- as.character(sample_sizes)
  
  # Loop through each variable
  row_counter <- 2
  for (var in vars) {
    if (is.factor(df[, var]) | is.character(df[, var])) {
      # Add the factor name
      row_names <- c(row_names, var)
      results[[row_counter]] <- rep("", length(df_split))
      row_counter <- row_counter + 1
      
      factor_levels <- unique(df[, var])
      factor_levels <- factor_levels[!is.na(factor_levels)]
      
      if (factor_rows) {
        # Add a row for the factor levels with indentation
        row_names <- c(row_names, paste0("    ", paste(factor_levels, collapse = "/")))
        results[[row_counter]] <- rep("", length(df_split))
        row_counter <- row_counter + 1
        
        for (j in 1:length(df_split)) {
          x <- df_split[[j]]
          results[[row_counter]] <- format_factor(x[, var])
        }
        row_counter <- row_counter + 1
      } else {
        # Add a row for each level and its percentage
        for (level in factor_levels) {
          row_names <- c(row_names, paste0("    ", level))  # Indent factor level names
          result_row <- c()
          for (j in 1:length(df_split)) {
            x <- df_split[[j]]
            level_freq <- sum(x[, var] == level, na.rm = TRUE)
            level_percent <- round((level_freq / nrow(x)) * 100, 2)
            result_row <- c(result_row, paste0(level_percent, "%"))
          }
          results[[row_counter]] <- result_row
          row_counter <- row_counter + 1
        }
      }
    } else {
      # For non-factor variables, just add one row
      row_names <- c(row_names, var)
      result_row <- c()
      for (j in 1:length(df_split)) {
        x <- df_split[[j]]
        result_row <- c(result_row, format_output(x[, var]))
      }
      results[[row_counter]] <- result_row
      row_counter <- row_counter + 1
    }
  }
  
  # Convert results to a data frame
  df_results <- as.data.frame(do.call(rbind, results))
  
  # Set row names
  rownames(df_results) <- row_names
  
  # Format column names with sample sizes
  if (!is.null(col_names)) {
    colnames(df_results) <- paste0(col_names, " (N=", sample_sizes, ")")
  } else {
    # Generate default column names with sample sizes
    col_names <- paste0(names(df_split))
    colnames(df_results) <- paste0(col_names, " (N=", sample_sizes, ")")
  }
  
  # Return the data frame
  return(df_results[-1,])
}

#' Generate a table summarizing the data in a linear model formula.
#'
#' @param formula Model formula.
#' @param data Data frame containing the data.
#' @param facet_var Name of the facet variable.
#'
#' @return A data frame summarizing the data in the formula.
#'
#' @examples
#' \dontrun{
#' table_1_from_formula(formula, data, facet_var)
#' }
#' @export
table_1_from_formula <- function(formula, data, facet_var) {
  # Extract the variables from the formula
  terms <- terms(formula)
  vars <- attr(terms, "term.labels")
  
  # Remove random effects
  vars <- vars[!grepl("\\|", vars)]
  
  # Remove the facet variable
  vars <- setdiff(vars, facet_var)
  
  # Call table_1_facet to summarize the data
  table_1(data, vars, facet_var,factor_rows=FALSE)
}



#' Format a table for publication using LaTeX or html
#'
#' @param df Data frame containing the table data.
#' @param caption Caption for the table.
#' @param label Label for the table.
#' @param html_font character eg Arial
#'
#' @return A LaTeX-formatted table.
#'
#' @examples
#' \dontrun{
#' table_1_presentation(df, caption = "Demographic and Clinical Characteristics",
#'                          label = "table:demographics")
#' }
#' @export
table_1_presentation <- function(df, caption = "", label = "", format = "latex", html_font='Arial') {
  if (!is.data.frame(df)) {
    stop("Input must be a data frame")
  }
  
  # Check if required libraries are installed and loaded
  if (!requireNamespace("knitr") || !requireNamespace("kableExtra")) {
    stop("Required libraries not installed or loaded. Please install and load 'knitr' and 'kableExtra'.")
  }
  # Load necessary libraries
  library(knitr)
  library(kableExtra)
  
  # Format the table
  if (format == "latex") {
    df %>%
      kable(format = "latex", booktabs = TRUE, 
            caption = caption, label = label) %>%
      kable_styling(latex_options = c("striped", "hold_position")) %>%
      column_spec(1, width = "3cm") %>%
      row_spec(0, bold = TRUE, font_size = 12) %>%
      row_spec(1, italic = TRUE, font_size = 10)
  } else if (format == "html") {
    df %>%
      kable(format = "html", caption = caption) %>%
      kable_styling(bootstrap_options = c("striped", "hover"), html_font = html_font ) %>%
      row_spec(0, bold = TRUE)
  }
}



#' Count Unique Subjects per Diagnosis
#'
#' This function counts the number of unique subjects for each diagnosis in a dataframe.
#' If `format_counts` is `TRUE`, it returns a formatted string of counts.
#'
#' @param df A dataframe containing the data.
#' @param diagnosis_col A string specifying the column name for diagnoses.
#' @param subject_col A string specifying the column name for subjects.
#' @param format_counts A boolean indicating whether to format counts as a single string
#'   in the format "x/y/z/q" where x, y, z, q represent the counts for each diagnosis. Defaults to FALSE.
#'
#' @return A dataframe with two columns: Diagnosis and Unique_Subjects. If `format_counts` is TRUE,
#'   only the formatted counts are returned.
#'
#' @examples
#' \dontrun{
#' df <- data.frame(
#'   diagnosis = c('A', 'A', 'B', 'A', 'B', 'C', 'C', 'C'),
#'   subject_id = c(1, 2, 2, 3, 4, 5, 6, 6)
#' )
#' count_unique_subjects_per_diagnosis(df, 'diagnosis', 'subject_id')
#' count_unique_subjects_per_diagnosis(df, 'diagnosis', 'subject_id', format_counts = TRUE)
#' }
#' @export
count_unique_subjects_per_diagnosis <- function(df, diagnosis_col, subject_col, format_counts = FALSE) {
  # Ensure the input columns are present in the dataframe
  if (!(diagnosis_col %in% names(df)) | !(subject_col %in% names(df))) {
    stop("The specified columns are not present in the dataframe.")
  }
  
  # Group by diagnosis and count unique subjects
  result <- df %>%
    dplyr::group_by(dplyr::across(all_of(diagnosis_col))) %>%
    dplyr::summarise(Unique_Subjects = dplyr::n_distinct(.data[[subject_col]]), .groups = 'drop') %>%
    dplyr::rename(Diagnosis = !!sym(diagnosis_col)) %>%
    dplyr::arrange(Diagnosis)
  
  # Return only formatted counts if format_counts is TRUE
  if (format_counts) {
    formatted_counts <- paste(result$Unique_Subjects, collapse = "/")
    return(data.frame(Unique_Subjects = formatted_counts))
  }
  
  # Return the result with both Diagnosis and Unique_Subjects
  return(result)
}




#' Compute ASymmetry-Adjusted Mean (ASAM) for Asymmetry Analysis
#'
#' This function computes the Symmetry-Adjusted Mean (SAM) between two values, typically representing 
#' the left and right measurements of a structure. SAM is a combined metric that incorporates both 
#' the average value and the asymmetry between the two sides.
#'
#' The formula is: 
#' \deqn{ASAM = \frac{Left + Right}{2} - |Left - Right|}
#'
#' This metric captures both the overall size (average) and the degree of asymmetry between the two sides.
#' 
#' ## Thought Experiment Explanation:
#' Consider a comparison between two groups (e.g., Control and Disease) where you want to assess 
#' differences in both the size and symmetry of a structure, such as the left and right hippocampus:
#'
#' - In the **Control** group, where symmetry is expected, SAM will be close to the average of the two sides.
#' - In the **Disease** group, where asymmetry is expected, SAM will be lower due to the subtraction of the absolute difference between the left and right sides.
#'
#' This metric is insightful because it combines information about both the structure's size and symmetry, 
#' allowing it to reflect differences in both characteristics.
#'
#' @param left Numeric. The value representing the left side of the measurement.
#' @param right Numeric. The value representing the right side of the measurement.
#' @param weight Numeric. increase weight on asymmetry portion of equation.
#' 
#' @return Numeric value representing the Symmetry-Adjusted Mean (SAM).
#'
#' @examples
#' # Example usage:
#' left <- 5
#' right <- 3
#' asam(left, right)
#' 
#' # For a symmetric case:
#' asam(5, 5)
#'
#' @export
asam <- function(left, right, weight = 1) {
  # Compute the average of Left and Right
  averaged <- 0.5 * (left + right)
  
  # Compute the absolute difference between Left and Right
  asymmetry <- abs(left - right)
  
  # Compute the Symmetry-Adjusted Mean (SAM)
  sam <- averaged - asymmetry * weight
  
  return(sam)
}




#' Asymmetry index for asymmetry analysis
#'
#' The formula is: 
#' \deqn{asymind = \frac{2|A-B|}{(A+B)} }
#'
#' @param left Numeric. The value representing the left side of the measurement.
#' @param right Numeric. The value representing the right side of the measurement.
#' 
#' @return Numeric value representing the Symmetry-Adjusted Mean (SAM).
#'
#' @examples
#' # Example usage:
#' left <- 5
#' right <- 3
#' asymind(left, right)
#' 
#' # For a symmetric case:
#' asymind(5, 5)
#'
#' @export
asymind <- function(left, right) {
  # Compute the average of Left and Right
  averaged <- 0.5 * (left + right)
  
  # Compute the absolute difference between Left and Right
  asymmetry <- abs(left - right)
  
  # Compute the Symmetry-Adjusted Mean (SAM)
  sam <- asymmetry/averaged
  
  return(sam)
}



#' Compute Various Asymmetry Metrics Between Two Sides
#'
#' This function computes different types of asymmetry metrics given left and right side measurements.
#' It can return standard asymmetry indices, including the absolute difference, symmetry-adjusted mean,
#' proportional asymmetry, and more.
#'
#' @param left Numeric. The value representing the left side of the measurement.
#' @param right Numeric. The value representing the right side of the measurement.
#' @param name_of_measurement Character. The name of the asymmetry measure to compute. Options include:
#' \itemize{
#'   \item "absolute_difference": Absolute difference between left and right.
#'   \item "proportional_asymmetry": Proportional asymmetry.
#'   \item "symmetry_adjusted_mean": Symmetry-adjusted mean (SAM).
#'   \item "relative_asymmetry": (Left - Right) / (Left + Right).
#'   \item "asymmetry_index": (|Left - Right|) / ((Left + Right) / 2).
#' }
#'
#' @return Numeric value of the computed asymmetry measure.
#' @examples
#' asymmetry(5, 3, "absolute_difference")
#' asymmetry(10, 8, "symmetry_adjusted_mean")
#'
#' @export
asymmetry <- function(left, right, name_of_measurement) {
  
  # Compute different asymmetry metrics based on the user input
  switch(name_of_measurement,
         
         # Absolute difference
         "absolute_difference" = abs(left - right),
         
         # Proportional asymmetry: (Left - Right) / ((Left + Right) / 2)
         "proportional_asymmetry" = abs(left - right) / ((left + right) / 2),
         
         # Symmetry-Adjusted Mean (SAM): (Left + Right) / 2 - |Left - Right|
         "symmetry_adjusted_mean" = (left + right) / 2 - abs(left - right),
         
         # Relative asymmetry: (Left - Right) / (Left + Right)
         "relative_asymmetry" = (left - right) / (left + right),
         
         # Asymmetry index: |Left - Right| / ((Left + Right) / 2)
         "asymmetry_index" = abs(left - right) / ((left + right) / 2),
         
         stop("Invalid name_of_measurement. Choose from: 'absolute_difference', 'proportional_asymmetry', 'symmetry_adjusted_mean', 'relative_asymmetry', or 'asymmetry_index'.")
  )
}


#' Collect and Zip Images Based on Subject-Date List
#'
#' This function collects image files from subfolders based on a list of subject-date IDs 
#' and zips them into a single archive. The user specifies the root directory, 
#' the modality folder, and the file extension. The zip file name is also user-defined.
#'
#' @param subjectIDdate_list A list of subjectIDdate values, where each entry is in the format "subjectID-date" (e.g., "12345-20220101").
#' @param root_path The root directory where the subject and date folders are stored.
#' @param modality The modality folder to search for images (e.g., "MRI", "CT").
#' @param extension The file extension to search for (default is "png").
#' @param zip_filename The name of the output zip file (default is "images.zip").
#'
#' @return Invisibly returns the name of the zip file that was created.
#' If no images are found, the function stops with an error.
#'
#' @examples
#' \dontrun{
#' # Example usage:
#' subjectIDdate_list <- c("121771-20220818", "42444-20210507", "56789-20210429")
#' root_path <- "/path/to/images"
#' modality <- "MRI"
#' zip_file <- collect_and_zip_images(subjectIDdate_list, root_path, modality, "png", "collected_images.zip")
#' }
#'
#' @export
collect_and_zip_images <- function(subjectIDdate_list, root_path, modality, extension = "png", zip_filename = "images.zip") {
  # Function to collect images based on subjectIDdate and modality, then zip them into a single file
  # subjectIDdate_list: list of subjectIDdate values like "12345-20220101"
  # root_path: root directory where images are stored
  # modality: the modality folder to look into (e.g., "MRI", "CT", etc.)
  # extension: file extension (default is "png")
  # zip_filename: name of the output zip file
  
  # Helper function to parse subject ID and date
  parse_subject_date <- function(id_date) {
    parts <- unlist(strsplit(id_date, "-"))
    subject <- parts[1]
    date <- parts[2]
    return(list(subject = subject, date = date))
  }

  # Initialize a list to store found file paths
  image_paths <- list()

  # Iterate over the subjectIDdate list
  for (id_date in subjectIDdate_list) {
    parsed_info <- parse_subject_date(id_date)
    subject <- parsed_info$subject
    date <- parsed_info$date

    # Construct the full path to the images folder
    image_dir <- file.path(root_path, subject, date, modality, '*', '*' )

    # Collect all files with the chosen extension in that folder
    image_files = Sys.glob( paste0(image_dir, '*', extension) )

    # Add the found files to the list
    if (length(image_files) > 0) {
      image_paths[[id_date]] <- image_files
    } else {
      message(paste("No", extension, "files found for", id_date, "in", image_dir))
    }
  }

  # Flatten the list into a vector of file paths
  all_files <- unlist(image_paths)
  
  if (length(all_files) == 0) {
    stop("No files found to zip.")
  }

  # Create the zip archive
  zip(zipfile = zip_filename, files = all_files)

  message(paste("Files zipped into", zip_filename))
  return(invisible(zip_filename))
}



#' Create a Styled LaTeX Table with Adjustable Size and Orientation
#'
#' This function generates a LaTeX table from a data frame using the `kableExtra` package, 
#' with options to adjust the table size, apply striped styling, and set the table orientation 
#' to landscape or portrait mode.
#'
#' @param data A data frame to be displayed in the table.
#' @param caption A character string for the table caption.
#' @param scl A numeric value to scale the table size. Defaults to 0.75.
#' @param row.names Logical, whether to include row names in the table. Defaults to FALSE.
#' @param striped Logical, whether to apply striped styling to the table. Defaults to TRUE.
#' @param landscape Logical, whether to rotate the table to landscape mode. Defaults to TRUE.
#' @param table_size A character string for adjusting the overall size of the table. Options include "small", "medium", "large". Defaults to "medium".
#' @param digits An integer specifying the number of decimal places for numeric values. Defaults to 3.
#' @param latex_options A character vector for specifying additional LaTeX styling options. Defaults to `c("striped")`.
#' @param format either latex or html
#'
#' @return A LaTeX table generated with `kableExtra`, ready for inclusion in an RMarkdown or LaTeX document.
#' @examples
#' data <- mtcars[1:5, 1:5]
#' kable_table(data, caption = "Sample Table", scl = 0.8, row.names = TRUE, striped = TRUE, landscape = FALSE, table_size = "large")
#'
#' @export
kable_table <- function(data, caption, scl = 0.75, row.names = FALSE, striped = TRUE, landscape = TRUE, 
                        table_size = "medium", digits = 3, latex_options = c("striped"), format='latex') {
  
  library(kableExtra)
  
  # Define the table size based on input
  size_map <- list(small = "\\small", medium = "\\normalsize", large = "\\large")
  table_size_tag <- size_map[[table_size]]
  if (is.null(table_size_tag)) {
    stop("Invalid table_size argument. Choose from 'small', 'medium', or 'large'.")
  }
  

  # Create the basic LaTeX table
  table <- kable(data, format = format, caption = caption, booktabs = TRUE,
                 row.names = row.names, digits = digits, scale_down=scl ) %>%
#    add_header_above(c(table_size_tag)) %>%
    kable_styling(latex_options = latex_options, full_width = FALSE) %>%
      column_spec(1, bold = TRUE)
  
  # Apply striped styling if requested
  if (striped) {
    table <- table %>%
      kable_styling(latex_options = c("striped"), stripe_color = "gray!22")
  }
  
  
  # Rotate to landscape if requested
  if (landscape) {
    table <- table %>%
      landscape()
  }
  
  return(table)
}



#' Rank Rows Based on Weighted Scores of Specified Columns
#'
#' This function ranks rows in a data frame based on the weighted scores of selected columns.
#' The function normalizes the specified columns (using Min-Max normalization), computes a weighted score for each row, and assigns a rank based on these scores.
#' If no weights are provided, the function defaults to equal weighting for the columns.
#'
#' @param df A data frame containing the columns to be ranked.
#' @param columns_to_maximize A character vector of column names that should be maximized and used for ranking.
#' @param weights A numeric vector of weights for the columns. The length of this vector must match the number of columns in `columns_to_maximize`. Defaults to `NULL`, in which case equal weights are applied.
#' 
#' @return A data frame with the original columns, the normalized columns, the weighted score for each row, and the rank based on the weighted score.
#' 
#' @examples
#' # Create a sample data frame
#' df <- data.frame(
#'   A = c(1, 2, 3, 4),
#'   B = c(2, 3, 4, 5),
#'   C = c(3, 4, 5, 6)
#' )
#' 
#' # Specify columns to maximize
#' columns_to_maximize <- c("A", "B")
#' 
#' # Rank the rows based on the specified columns
#' ranked_df <- rank_results(df, columns_to_maximize, weights = c(0.7, 0.3))
#' 
#' # View the ranked data frame
#' print(ranked_df)
#' 
#' @export
rank_results <- function(df, columns_to_maximize, weights = NULL) {
  # Ensure the columns_to_maximize exist in the dataframe
  if (!all(columns_to_maximize %in% colnames(df))) {
    stop("Some columns_to_maximize do not exist in the dataframe.")
  }
  
  # Normalize the selected columns (Min-Max normalization)
  normalize <- function(x) {
    (x - min(x)) / (max(x) - min(x))
  }
  
  # Apply normalization to the columns that should be maximized
  normalized_df <- df %>%
    mutate(across(all_of(columns_to_maximize), normalize, .names = "norm_{col}"))
  
  # Prepare a list of normalized columns
  normalized_columns <- paste0("norm_", columns_to_maximize)
  
  # If no weights provided, default to equal weighting
  if (is.null(weights)) {
    weights <- rep(1, length(normalized_columns))
  }
  
  # Ensure weights match the number of columns
  if (length(weights) != length(normalized_columns)) {
    stop("Number of weights must match the number of columns to maximize.")
  }
  
  # Compute the weighted score for each row
  normalized_df <- normalized_df %>%
    rowwise() %>%
    mutate(weighted_score = sum(c_across(all_of(normalized_columns)) * weights)) %>%
    ungroup()
  
  # Rank the rows based on the weighted score (higher is better)
  normalized_df <- normalized_df %>%
    arrange(desc(weighted_score)) %>%
    mutate(rank = row_number())
  
  # Return the dataframe with rankings and scores
  return(normalized_df)
}


#' Subset Dataframe for Subjects with Multiple Unique Visits
#'
#' This function subsets a dataframe to return only subjects who have more than one unique visit.
#' It takes in the column names for the subject identifier and the visit identifier, then filters
#' the dataframe to include only rows corresponding to subjects with multiple distinct visits.
#'
#' @param data A dataframe containing the data to be subset.
#' @param subject_identifier A string specifying the column name used to identify subjects.
#' @param visit_identifier A string specifying the column name used to identify visits.
#'
#' @return A dataframe containing only subjects with more than one unique visit.
#'
#' @examples
#' # Example dataframe
#' df <- data.frame(
#'   subjectID = c(1, 1, 2, 2, 3, 3, 4),
#'   visitID = c(1, 2, 1, 1, 1, 3, 1)
#' )
#' 
#' # Subset for subjects with multiple unique visits
#' result <- subset_multiple_visits(df, "subjectID", "visitID")
#' print(result)
#'
#' @export
subset_multiple_visits <- function(data, subject_identifier, visit_identifier) {
  # Ensure the columns are in the dataframe
  if (!all(c(subject_identifier, visit_identifier) %in% names(data))) {
    stop("Specified columns not found in the dataframe.")
  }
  
  # Find subjects with more than one unique visit
  subjects_with_multiple_visits <- data %>%
    group_by_at(subject_identifier) %>%
    filter(n_distinct(!!sym(visit_identifier)) > 1) %>%
    ungroup()
  
  return(subjects_with_multiple_visits)
}




#' Harmonize Multiple Features Across Sites with Progress Indicator
#'
#' Adjusts specified features across sites so that the control group within each site
#' matches the mean of the control group in a reference site. The transformation is applied
#' to all data within each site, regardless of diagnosis, for each feature separately.
#'
#' @param data A data frame containing the data.
#' @param site_col A string indicating the column name for site identifiers.
#' @param diagnosis_col A string indicating the column name for diagnosis identifiers.
#' @param control_label The label in the diagnosis column identifying the control group.
#' @param feature_cols A vector of strings specifying the feature columns to be harmonized.
#' @param reference_site The site identifier to use as the reference site for control means.
#'
#' @return A list containing:
#'   - `harmonized_data`: the data frame with features adjusted across sites
#'   - `summary_stats`: a data frame with original control means by site and reference means for each feature
#' @examples
#' harmonize_sites(df, site_col = "Site", diagnosis_col = "Diagnosis", 
#'                 control_label = "Control", feature_cols = c("Feature1", "Feature2"), reference_site = "SiteA")
harmonize_sites <- function(data, site_col, diagnosis_col, control_label, feature_cols, reference_site) {
  # Check if specified columns exist
  if (!all(c(site_col, diagnosis_col, feature_cols) %in% colnames(data))) {
    stop("One or more specified columns do not exist in the data.")
  }
  
  # Filter control group data in the reference site and calculate the reference means for each feature
  ref_control_data <- data[data[[site_col]] == reference_site & data[[diagnosis_col]] == control_label, ]
  if (nrow(ref_control_data) == 0) stop("No control data found for the specified reference site.")
  ref_means <- sapply(feature_cols, function(col) mean(ref_control_data[[col]], na.rm = TRUE))
  
  # Initialize a data frame to store original control means by site for each feature
  summary_stats <- data.frame(Site = unique(data[[site_col]]))
  for (col in feature_cols) {
    summary_stats[[paste0("Original_Control_Mean_", col)]] <- NA
  }
  summary_stats <- cbind(summary_stats, Reference_Control_Mean = ref_means)
  
  # Harmonized data frame to store transformed data
  harmonized_data <- data
  
  # Set up progress bar
  total_tasks <- length(unique(data[[site_col]])) * length(feature_cols)
  pb <- txtProgressBar(min = 0, max = total_tasks, style = 3)
  task_count <- 0
  
  # Adjust each feature column separately by site
  for (current_site in unique(data[[site_col]])) {
    # Filter data for the current site
    site_data <- harmonized_data[harmonized_data[[site_col]] == current_site, ]
    site_controls <- site_data[site_data[[diagnosis_col]] == control_label, ]
    
    # Check if site has control data
    if (nrow(site_controls) > 0) {
      for (feature in feature_cols) {
        # Calculate the site-specific mean for the control group for the current feature
        site_mean <- mean(site_controls[[feature]], na.rm = TRUE)
        # Store the original control mean for the feature in summary_stats
        summary_stats[summary_stats$Site == current_site, paste0("Original_Control_Mean_", feature)] <- site_mean
        
        # Apply transformation to all subjects in the site for this feature
        harmonized_data[harmonized_data[[site_col]] == current_site, feature] <-
          harmonized_data[harmonized_data[[site_col]] == current_site, feature] + (ref_means[feature] - site_mean)
        
        # Update progress bar
        task_count <- task_count + 1
        setTxtProgressBar(pb, task_count)
      }
    } else {
      warning("No control data for site ", current_site, ". Skipping adjustment for this site.")
    }
  }
  
  # Close progress bar
  close(pb)
  
  return(list(harmonized_data = harmonized_data, summary_stats = summary_stats))
}


#' Harmonize Multiple Features Across Sites with Quantile Matching
#'
#' Adjusts specified features across sites to match the quantiles (e.g., 25th, 50th, and 75th) of the control group
#' in the reference site. The transformation is applied to all data within each site, regardless of diagnosis, for each feature separately.
#'
#' @param data A data frame containing the data.
#' @param site_col A string indicating the column name for site identifiers.
#' @param diagnosis_col A string indicating the column name for diagnosis identifiers.
#' @param control_label The label in the diagnosis column identifying the control group.
#' @param feature_cols A vector of strings specifying the feature columns to be harmonized.
#' @param reference_site The site identifier to use as the reference site for control quantiles.
#'
#' @return A list containing:
#'   - `harmonized_data`: the data frame with features adjusted across sites.
#'   - `summary_stats`: a data frame with quantiles (e.g., 25th, 50th, and 75th) by site for each feature.
#' @export
#' @examples
#' harmonize_sites_quantiles(df, site_col = "Site", diagnosis_col = "Diagnosis", 
#'                           control_label = "Control", feature_cols = c("Feature1", "Feature2"), 
#'                           reference_site = "SiteA")
harmonize_sites_quantiles <- function(data, site_col, diagnosis_col, control_label, feature_cols, reference_site) {
  # Check if specified columns exist
  if (!all(c(site_col, diagnosis_col, feature_cols) %in% colnames(data))) {
    stop("One or more specified columns do not exist in the data.")
  }
  
  # Filter control group data in the reference site and calculate the quantiles (25th, 50th, 75th) for each feature
  ref_control_data <- data[data[[site_col]] == reference_site & data[[diagnosis_col]] == control_label, ]
  if (nrow(ref_control_data) == 0) stop("No control data found for the specified reference site.")
  
  # Calculate quantiles (25th, 50th, 75th) for each feature in the reference site's control group
  ref_quantiles <- sapply(feature_cols, function(col) {
    quantile(ref_control_data[[col]], probs = c(0.25, 0.5, 0.75), na.rm = TRUE)
  })
  
  # Initialize a data frame to store original control quantiles by site for each feature
  summary_stats <- data.frame(Site = unique(data[[site_col]]))
  for (col in feature_cols) {
    summary_stats[[paste0("Original_Control_25th_", col)]] <- NA
    summary_stats[[paste0("Original_Control_50th_", col)]] <- NA
    summary_stats[[paste0("Original_Control_75th_", col)]] <- NA
  }
  summary_stats <- cbind(summary_stats, Reference_Control_25th = ref_quantiles[1,],
                         Reference_Control_50th = ref_quantiles[2,],
                         Reference_Control_75th = ref_quantiles[3,])
  
  # Harmonized data frame to store transformed data
  harmonized_data <- data
  
  # Set up progress bar
  total_tasks <- length(unique(data[[site_col]])) * length(feature_cols)
  pb <- txtProgressBar(min = 0, max = total_tasks, style = 3)
  task_count <- 0
  
  # Adjust each feature column separately by site
  for (current_site in unique(data[[site_col]])) {
    # Filter data for the current site
    site_data <- harmonized_data[harmonized_data[[site_col]] == current_site, ]
    site_controls <- site_data[site_data[[diagnosis_col]] == control_label, ]
    
    # Check if site has control data
    if (nrow(site_controls) > 0) {
      for (feature in feature_cols) {
        # Calculate the quantiles for the control group in the current site for the feature
        site_quantiles <- quantile(site_controls[[feature]], probs = c(0.25, 0.5, 0.75), na.rm = TRUE)
        
        # Store the original control quantiles for the feature in summary_stats
        summary_stats[summary_stats$Site == current_site, paste0("Original_Control_25th_", feature)] <- site_quantiles[1]
        summary_stats[summary_stats$Site == current_site, paste0("Original_Control_50th_", feature)] <- site_quantiles[2]
        summary_stats[summary_stats$Site == current_site, paste0("Original_Control_75th_", feature)] <- site_quantiles[3]
        
        # Apply transformation to align quantiles (25th, 50th, and 75th) with the reference site
        harmonized_data[harmonized_data[[site_col]] == current_site, feature] <-
          (harmonized_data[harmonized_data[[site_col]] == current_site, feature] - site_quantiles[2]) / (site_quantiles[3] - site_quantiles[1]) * (ref_quantiles[3, feature] - ref_quantiles[1, feature]) + ref_quantiles[2, feature]
        
        # Update progress bar
        task_count <- task_count + 1
        setTxtProgressBar(pb, task_count)
      }
    } else {
      warning("No control data for site ", current_site, ". Skipping adjustment for this site.")
    }
  }
  
  # Close progress bar
  close(pb)
  
  return(list(harmonized_data = harmonized_data, summary_stats = summary_stats))
}



#' Apply ComBat Batch Correction While Preserving Missing Data
#'
#' This function performs batch effect correction using the ComBat method from the \code{sva} package. It applies ComBat to the data after imputing missing values, but preserves the missing data (NA values) in the output.
#'
#' @param df A numeric matrix or data frame with missing values (NA) that you want to adjust for batch effects.
#' @param batch A numeric or character vector containing the batch assignments (must have the same length as the number of samples in \code{df}).
#' 
#' @return A numeric matrix or data frame with batch effects corrected, while preserving the original NA values from \code{df}.
#'
#' @examples
#' # Assuming 'df' is your data frame with missing values and 'batch' is a vector of batch assignments
#' adjusted_data <- combat_with_na(df, batch)
#'
#' @import sva
#' @export
combat_with_na <- function(df, batch) {
  # Check if the length of batch matches the number of columns in df
  if (length(batch) != nrow(df)) {
    stop("The length of batch must match the number of columns in df.")
  }
  
  # Step 1: Impute the data and transpose it
  imputed_data <- t(antsrimpute(df))  # Transpose after imputation
  
  # Step 2: Identify the locations of the original NAs in the df data
  na_indices <- is.na(df)
  
  # Step 3: Apply ComBat to the imputed data
  cbt <- sva::ComBat(dat = imputed_data, batch = as.numeric(batch))
  
  # Step 4: Convert the ComBat output back to the same shape as the original data
  cbt_transposed <- t(cbt)  # Transpose back to match original data shape
  
  # Step 5: Preserve NAs from the original data by setting those values back to NA
  cbt_transposed[na_indices] <- NA
  
  # Return the ComBat-adjusted data with NAs preserved
  return(cbt_transposed)
}