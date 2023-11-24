#!/bin/bash

table=$1
clickhouse client --query "drop table xip.$table"
clickhouse client --query "show tables in xip"