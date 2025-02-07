#' Impute Missing Values and Report NA Counts
#'
#' This function imputes missing values in a matrix with column means and reports the count of missing values per column.
#'
#' @param mat A numeric matrix.
#' @return A matrix with imputed values.
#' @examples
#' mat <- matrix(c(1, NA, 3, 4, 5, NA), ncol = 2)
#' impute_and_report_na(mat)
#' @export
impute_and_report_na <- function(mat) {
  if (!is.matrix(mat)) {
    stop("Input must be a matrix.")
  }
  col_names <- colnames(mat)
  na_report <- list()
  for (i in seq_along(col_names)) {
    col_data <- mat[, i]
    na_count <- sum(is.na(col_data))
    if (na_count > 0) {
      col_mean <- mean(col_data, na.rm = TRUE)
      col_data[is.na(col_data)] <- col_mean
      na_report[[col_names[i]]] <- na_count
    }
    mat[, i] <- col_data
  }
  if (length(na_report) > 0) {
    cat("NA counts per column:\n")
    print(na_report)
  } else {
    cat("No missing values detected.\n")
  }
  return(mat)
}

#' Convert Data Frame to Numeric Matrix
#'
#' Converts a data frame with categorical and numeric columns into a numeric matrix, using one-hot encoding for categorical variables.
#'
#' @param df A data frame.
#' @return A numeric matrix.
#' @examples
#' df <- data.frame(a = c(1, 2), b = c("yes", "no"))
#' convert_to_numeric_matrix(df)
#' @export
convert_to_numeric_matrix <- function(df) {
  if (!requireNamespace("fastDummies", quietly = TRUE)) {
    install.packages("fastDummies")
  }
  library(fastDummies)
  cat_cols <- sapply(df, is.character) | sapply(df, is.factor)
  num_cols <- !cat_cols
  if (all(num_cols)) {
    return(as.matrix(df))
  }
  if (any(cat_cols)) {
    df_cat <- fastDummies::dummy_cols(df[, cat_cols, drop = FALSE], 
                                      remove_first_dummy = FALSE, 
                                      remove_selected_columns = TRUE)
  } else {
    df_cat <- NULL
  }
  df_num <- if (any(num_cols)) df[, num_cols, drop = FALSE] else NULL
  df_final <- cbind(df_num, df_cat)
  return(as.matrix(df_final))
}

#' Count Unique Values and Filter Columns
#'
#' Counts unique values in each column of a data frame and retains columns exceeding a specified threshold.
#'
#' @param df A data frame.
#' @param threshold An integer specifying the minimum number of unique values to retain a column.
#' @return A data frame with filtered columns.
#' @examples
#' df <- data.frame(a = c(1, 1, 2), b = c("x", "y", "z"))
#' count_unique_values(df, 1)
#' @export
count_unique_values <- function(df, threshold) {
  results <- logical(ncol(df))
  names(results) <- colnames(df)
  for (col_name in colnames(df)) {
    unique_count <- length(unique(df[[col_name]]))
    cat(col_name, ":", unique_count, "unique values\n")
    results[col_name] <- unique_count > threshold
  }
  return(df[, results])
}




#' Read and Filter XPT File
#'
#' Reads an XPT file, converts it to a data frame, renames the first column to "ID", and filters based on unique value counts.
#'
#' @param file_path Path to the XPT file.
#' @param threshold Numeric threshold to filter columns based on unique value counts. Default is 100.
#'
#' @return A data frame with filtered columns.
#' @examples
#' read_and_filter_xpt("data.xpt", threshold = 50)
#' @export
read_and_filter_xpt <- function(file_path, threshold = 100) {
  df <- data.frame(read_xpt(file_path))
  colnames(df)[1] <- "ID"
  return(count_unique_values(df, threshold))
}

#' Impute Missing Data
#'
#' Imputes missing values in a data frame. Numeric columns are imputed with the median, and categorical columns with the most frequent category.
#'
#' @param df A data frame with missing values.
#'
#' @return A data frame with imputed values.
#' @examples
#' impute_data(data.frame(a = c(1, NA, 3), b = c("x", "y", NA)))
#' @export
impute_data <- function(df) {
  imputed_df <- df  # Copy input data
  
  for (col in names(df)) {
    if (is.numeric(df[[col]])) {
      median_val <- median(df[[col]], na.rm = TRUE)
      imputed_df[[col]][is.na(imputed_df[[col]])] <- median_val
    } else {
      freq_table <- table(df[[col]], useNA = "no")
      most_frequent <- names(freq_table)[which.max(freq_table)]
      imputed_df[[col]][is.na(imputed_df[[col]])] <- most_frequent
    }
  }
  
  return(imputed_df)
}

#' Map Frequency Categories to Numeric Values
#'
#' Maps specific frequency categories to numeric values, with error handling for invalid categories.
#'
#' @param df A data frame containing the column to be mapped.
#' @param colname The name of the column to be mapped.
#'
#' @return A numeric vector with mapped values.
#' @examples
#' map_freq_to_numeric(data.frame(freq = c("Not at all", "Several days")), "freq")
#' @export
map_freq_to_numeric <- function(df, colname) {
  if (!(colname %in% names(df))) {
    stop(paste("Column", colname, "does not exist in the dataframe"))
  }
  
  valid_categories <- c("Not at all", "Several days", "More than half the days", "Nearly every day", "Refused", "Don't know", NA)
  unique_values <- unique(df[[colname]])
  invalid_values <- setdiff(unique_values, valid_categories)

  if (length(invalid_values) > 0) {
    stop(paste("Invalid values in", colname, ":", paste(invalid_values, collapse = ", ")))
  }
  
  mapped_values <- dplyr::recode(df[[colname]],
                                 "Not at all" = 0,
                                 "Several days" = 1,
                                 "More than half the days" = 2,
                                 "Nearly every day" = 3,
                                 "Refused" = NA_real_,
                                 "Don't know" = NA_real_)
  return(mapped_values)
}

#' Merge Columns of the Same Type from Two Data Frames
#'
#' Merges columns of the same type from two data frames based on common column names.
#'
#' @param df1 First data frame.
#' @param df2 Second data frame.
#'
#' @return A merged data frame containing matching columns.
#' @examples
#' merge_same_type_columns(data.frame(a = 1:3), data.frame(a = 4:6))
#' @export
merge_same_type_columns <- function(df1, df2) {
  common_columns <- intersect(names(df1), names(df2))
  matching_columns <- common_columns[sapply(common_columns, function(col) {
    identical(class(df1[[col]]), class(df2[[col]]))
  })]

  merged_data <- bind_rows(df1[, matching_columns, drop = FALSE], df2[, matching_columns, drop = FALSE])
  return(merged_data)
}

#' Create Regression Formula
#'
#' Creates a regression formula from a data frame, excluding specified columns.
#'
#' @param dataframe The input data frame.
#' @param outcome The outcome variable.
#' @param exclusions A vector of columns to exclude from the formula.
#'
#' @return A formula object for regression.
#' @examples
#' create_formula(mtcars, "mpg", exclusions = c("cyl", "gear"))
#' @export
create_formula <- function(dataframe, outcome, exclusions) {
  all_columns <- colnames(dataframe)
  predictor_columns <- setdiff(all_columns, c(outcome, exclusions))

  valid_columns <- predictor_columns[sapply(predictor_columns, function(col) {
    if (is.numeric(dataframe[[col]]) && var(dataframe[[col]], na.rm = TRUE) == 0) return(FALSE)
    if (is.factor(dataframe[[col]]) && length(unique(dataframe[[col]])) == 1) return(FALSE)
    return(TRUE)
  })]

  formula <- as.formula(paste(outcome, "~", paste(valid_columns, collapse = " + ")))
  return(formula)
}

#' Check Formula Factors
#'
#' Identifies factor variables in a formula that have fewer than two levels.
#'
#' @param formula A formula object.
#' @param dataframe A data frame containing the variables in the formula.
#'
#' @return A vector of factor variable names with fewer than two levels.
#' @examples
#' check_formula_factors(mpg ~ cyl + gear, mtcars)
#' @export
check_formula_factors <- function(formula, dataframe) {
  terms <- all.vars(formula)[-1]
  problematic_terms <- terms[sapply(terms, function(term) {
    if (term %in% colnames(dataframe)) {
      col_data <- dataframe[[term]]
      return(is.factor(col_data) && length(unique(na.omit(col_data))) < 2)
    }
    return(FALSE)
  })]
  return(problematic_terms)
}

#' Filter Columns with High NA Percentage
#'
#' Filters out columns from a data frame that have a higher percentage of NA values than the specified threshold.
#'
#' @param df A data frame.
#' @param max_na_percent Maximum allowed proportion of NA values per column. Default is 0.2.
#'
#' @return A data frame with filtered columns.
#' @examples
#' filter_na_columns(data.frame(a = c(1, NA, 3), b = c(NA, NA, 3)), max_na_percent = 0.5)
#' @export
filter_na_columns <- function(df, max_na_percent = 0.2) {
  na_percent <- sapply(df, function(col) mean(is.na(col)))
  filtered_df <- df[, names(na_percent[na_percent <= max_na_percent]), drop = FALSE]
  return(filtered_df)
}




#' Generate a Matrix from a Latent Matrix with Added Noise
#'
#' This function generates a new matrix by applying a random linear transformation
#' to a given latent matrix and adding Gaussian noise.
#'
#' @param latent_matrix A numeric matrix representing the latent structure (n x k).
#' @param target_p An integer specifying the number of columns in the output matrix.
#' @param noise_sd A numeric value indicating the standard deviation of the Gaussian noise to be added. Default is 0.3.
#'
#' @return A numeric matrix of dimensions (n x target_p), generated from the latent matrix
#' with a random linear transformation and added Gaussian noise.
#'
#' @examples
#' latent <- matrix(rnorm(20), nrow = 5, ncol = 4)
#' generated_matrix <- matrix_from_latent(latent, target_p = 3, noise_sd = 0.2)
#' print(generated_matrix)
#'
#' @export
matrix_from_latent <- function(latent_matrix, target_p, noise_sd = 0.3) {
  # Get dimensions of latent matrix
  n <- nrow(latent_matrix)
  k <- ncol(latent_matrix)

  # Generate a random transformation matrix (k x target_p)
  transformation_matrix <- matrix(rnorm(k * target_p), nrow = k, ncol = target_p)

  # Generate the new matrix by multiplying latent matrix with transformation and adding noise
  new_matrix <- latent_matrix %*% transformation_matrix + 
                matrix(rnorm(n * target_p, sd = noise_sd), nrow = n, ncol = target_p)

  return(new_matrix)
}



#'
#' @format Curated nhanes cognition data from CERAD
#' @examples
#' \dontrun{
#'  nhanescog
#' }
"nhanescog"
