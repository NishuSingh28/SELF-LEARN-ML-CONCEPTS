---
layout: default
title: "CSV FILE Reading"
date: 2025-09-25
categories: [machine-learning]
---

# Working with CSV Files in Python using Pandas

## Introduction to CSV Files

- **CSV (Comma-Separated Values)** is the most common data format in machine learning (~90% usage).
- Data is stored in tabular form, with rows separated by newlines and values separated by commas.
- **TSV (Tab-Separated Values)** is similar but uses tabs instead of commas.

## Core Function: `pd.read_csv()`

`pd.read_csv()` is the main function used to read CSV files in Pandas.  
It supports numerous parameters to handle various real-world data scenarios.

## Key Parameters and Their Uses

### 1. `filepath_or_buffer`
- **Purpose:** File location.
- **Usage:**  
  - Local file → `'data.csv'`  
  - From URL → Use `requests` to fetch content before reading.

### 2. `sep` or `delimiter`
- **Purpose:** Defines separator between values.  
- **Default:** `,` (comma)  
- **For TSV:** `sep='\t'`

### 3. `names`
- **Purpose:** Custom column names when no header exists.  
- **Usage:** `names=['SerialNo', 'MovieName', 'ReleaseYear']`

### 4. `header`
- **Purpose:** Row used as column names.  
- **Default:** `header=0`  
- **No header:** `header=None`  
- **Custom row:** `header=1`

### 5. `usecols`
- **Purpose:** Load only specific columns.  
- **Usage:** `usecols=['Gender', 'EducationLevel']`

### 6. `squeeze`
- **Purpose:** Return a Series instead of DataFrame if only one column.  
- **Usage:** `squeeze=True`

### 7. `skiprows`
- **Purpose:** Skip initial rows.  
- **Usage:**  
  - Skip first 2 rows → `skiprows=2`  
  - Skip specific rows → `skiprows=[0, 2]`

### 8. `nrows`
- **Purpose:** Read only the first N rows.  
- **Usage:** `nrows=1000`

### 9. `encoding`
- **Purpose:** File encoding.  
- **Default:** `'utf-8'`  
- **Alternate encodings:** `'latin1'`, `'cp1252'` (useful for non-English text)

### 10. `error_bad_lines` / `on_bad_lines`
- **Purpose:** Handle malformed lines.  
- **Usage:**  
  - Skip bad lines → `on_bad_lines='skip'`

### 11. `dtype`
- **Purpose:** Define data type per column.  
- **Usage:** `dtype={'Target': 'int32'}`

### 12. `parse_dates`
- **Purpose:** Parse columns as datetime objects.  
- **Usage:** `parse_dates=['Date']`

### 13. `converters`
- **Purpose:** Apply custom conversion to columns.  
- **Usage:**  
  ```python
  converters={'Team': lambda x: x.replace('Royal Challengers Bangalore', 'RCB')}```

### 14. `na_values`

- **Purpose:** Define custom missing value indicators.
- **Usage:** na_values=['M', '-']

### 15. `chunksize`
- **Purpose:** Read large files in chunks.
- **Usage:** chunksize=5000
Use Case: Iterate over chunks to process data without loading entire file into memory.
