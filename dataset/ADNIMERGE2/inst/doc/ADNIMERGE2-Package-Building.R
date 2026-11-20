## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  warning = FALSE,
  message = FALSE,
  echo = FALSE
)

## ----dataset-list, echo = FALSE-----------------------------------------------
# Libraries
library(tidyverse)
library(ADNIMERGE2)
library(DT)

source(system.file("dataset-list.R", package = "ADNIMERGE2", mustWork = TRUE))

dataset_list <- bind_rows(
  # Data used in vignettes
  get_required_dataset_list(
    use_type = "article",
    add_url_link = TRUE
  ) %>%
    mutate(source_list = article_list),
  # Data used in data-preparation
  get_required_dataset_list(
    use_type = "prep_script",
    add_url_link = TRUE
  ) %>%
    mutate(source_list = script_list)
) %>%
  group_by(data_code) %>%
  mutate(source_list = toString(unique(source_list))) %>%
  ungroup() %>%
  # Add reference link
  concat_dataset_url(.data = ., var_name = "data_code") %>%
  select(
    `Dataset Code` = data_code,
    `Dataset Label` = label,
    `Use Articles/Scripts` = source_list,
    `Derived Dataset` = source_derived_data
  ) %>%
  distinct() %>%
  assert_uniq(., `Dataset Code`) %>%
  DT::datatable(.,
    options = list(paging = TRUE, searchable = TRUE, bInfo = FALSE),
    escape = FALSE,
    caption = "List of datasets required to replicate ADNIMERGE2 R data package"
  )
dataset_list

