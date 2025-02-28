% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/simlr_nhanes.R
\name{merge_same_type_columns}
\alias{merge_same_type_columns}
\title{Merge Columns of the Same Type from Two Data Frames}
\usage{
merge_same_type_columns(df1, df2)
}
\arguments{
\item{df1}{First data frame.}

\item{df2}{Second data frame.}
}
\value{
A merged data frame containing matching columns.
}
\description{
Merges columns of the same type from two data frames based on common column names.
}
\examples{
merge_same_type_columns(data.frame(a = 1:3), data.frame(a = 4:6))
}
