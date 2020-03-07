from __future__ import annotations
from .database_connector import DatabaseConnector
import sqlite3
import re
from pathlib import Path


class SQLite3Connector(DatabaseConnector):

    _connection: sqlite3.Connection = None

    def __init__(self, conn):
        self._connection = conn

    @classmethod
    def connect(cls, db_file: Path) -> SQLite3Connector:
        """
        Connects to sqlite3 database and returns connector.

        Args:
            db_file:    path to sqlite3 database file

        Returns:
            returns sqlite3 connector

        """

        if not db_file.parent.is_dir():
            raise NotADirectoryError(
                'sqlite3 database path must have valid parent directory.')

        try:
            conn = sqlite3.connect(db_file)

        except:
            raise e

        else:
            #            print(f'Connected to sqlite3 database at {db_file}.')
            return cls(conn)

    def close(self):
        assert(self._connection)
        self._connection.close()
#        print(f'Closed sliqte3 database.')

    def get_tables(self) -> List[str]:
        """
        Get list of tables.

        Returns:
            list of sqlite3 tables

        """

        query = 'SELECT name from sqlite_master where type= "table"'
        try:
            cursor = self._connection.cursor()
            result = cursor.execute(query)

        except sqlite3.Error as e:
            raise e

        else:
            tables = list()
            for r in result:
                tables.append(*r)

            return tables

    def table_exists(self, name: str) -> bool:
        """
        Checks if table exists.

        Args:
            name:   name of sqlite3 table

        Returns:
            bool indicating if table exists

        """

        try:
            cursor = self._connection.cursor()
            query = """SELECT * FROM sqlite_master
            WHERE type='table' and NAME=?;"""
            result = cursor.execute(query, [name]).fetchall()

        except sqlite3.Error as e:
            raise e

        else:
            return len(result) > 0

    def get_schema(self, table_name: str) -> List[Dict[str, Union[str, int, bool, float]]]:
        """
        Get schema of table.

        Args:
            table_name:     name of table

        Returns:
            list containing dictionaries of table columns

        """

        # Ensure table exists.
        if self.table_exists(table_name):

            # Precompile regex used to parse type and size; example: CHAR(32).
            size_regex = re.compile(
                '^(CHAR|VCHAR)(\(([1-9]\d*)\))$', re.IGNORECASE)

            # Query sqlite3 database using 'table_info'.
            try:
                cursor = self._connection.cursor()
                query = f'PRAGMA table_info("{table_name}");'
                result = cursor.execute(query)
            except sqlite3.Error as e:
                raise e

            schema = list()

            # Loop through rows in schema; each row corresponds to column of table.
            for row in result:

                # Unpack query result.
                cid, name, dtype, not_null, default, pk = (*row,)

                # Parse column ID to int.
                cid = int(cid)

                # Parse dtype into type and size using regex.
                size = None
                match = size_regex.match(dtype)

                if match is not None:
                    dtype = match.group(1).upper()

                    if match.group(3) is not None:
                        size = int(match.group(3))

                else:
                    dtype = dtype.upper()
                    valid_types = ('INT', 'INTEGER', 'CHAR',
                                   'DATE', 'VCHAR', 'FLOAT')
                    assert(valid_types.count(dtype) > 0)

                # Size was not specified for this column.
                if size is None:
                    if dtype == 'CHAR':
                        size = 1

                # Parse 'NOT NULL' field to boolean.
                not_null = bool(not_null)

                # Parse 'PRIMARY KEY' field to bolean.
                pk = bool(pk)

                # Append this column's properties to schema.
                schema.append({'cid': cid,
                               'name': name,
                               'dtype': dtype,
                               'size': size,
                               'not_null': not_null,
                               'default': default,
                               'pk': pk})

            return schema

        # Table was not found; raise exception.
        else:
            raise ValueError(f'Table {table_name} not found')

    def insert(self,
               table: str,
               w_columns: List[str],
               values: List[Tuple[Union[int, str, float], ...]]):
        """
        Inserts values for specified columns into table.

        Args:
            table:          name of table
            w_columns:      list of columns with which to insert
            values:         list of tuples containing values

        """

        # Create comma-seperated '?' for bindings.
        bindings = ','.join(('?' for _ in range(len(w_columns))))
#        print(f'bindings = {bindings}')
#        bindings = '(' + bindings + ')'

        # Construct sqlite3 query for INSERT.
        query = f'INSERT INTO {table} VALUES({bindings});'

#        print(query)

        try:
            # Execute query.
            cursor = self._connection.cursor()
            result = cursor.executemany(query, values).fetchall()

        except sqlite3.Error as e:
            raise(e)

    def select(self,
               table: str,
               w_columns: List[str] = None,
               w_filter: str = None) -> List[Tuple[Union[int, str, float], ...]]:
        """
        Gets data from database with specified columns and optional filter.

        Args:
            table:      name of table
            w_columns:  names of requested columns
            w_filter:   optional filter to apply to query

        Returns:
            list of tuples of requested entries

        """

        # Construct fitler 'WITH <x> = <y>' clause.
        filter_clause = w_filter if w_filter is not None else ''

        # Use wildcard to return all columns if no columns specified.
        if w_columns is None or len(w_columns) == 0:

            query = f'SELECT * FROM {table} {filter_clause};'

            try:
                cursor = self._connection.cursor()
                result = cursor.execute(query).fetchall()

            except sqlite3.Error as e:
                raise e
                return None

            else:
                return result

        # Columns were specified; return only selected columns.
        else:

            # Construct query with columns and possible filter clause.
            bindings = ','.join(w_columns)
            query = f'SELECT {bindings} FROM {table} {filter_clause};'

#            print(query)

            # Try to execute query and return records to caller.
            try:
                cursor = self._connection.cursor()
                result = cursor.execute(query).fetchall()

            except sqlite3.Error as e:
                raise e

            else:
                return result

    def delete(self, table: str, w_filter: str = None):
        """
        Deletes records from table with optional filter.

        Args:
            table:      name of table
            w_filter:   optional filter

        """

        # Construct filter clause and sqlite3 query.
        filter_clause = w_filter if w_filter is not None else ''
        query = f'DELETE FROM {table} {filter_clause}'

        # Try to execute query.
        try:
            cursor = self._connection.cursor()
            cursor.execute(query)
        except sqlite3.Error as e:
            raise e

    def create_table(self,
                     table_name: str,
                     columns: List[Dict[str, Union[int, str, bool, float]]]):
        """
        Creates table with specified columns.

        Args:
            table_name:         name of table
            columns:            list of dicts of column properties

        """

        column_strs = list()

        # Loop through list of column properites.
        for f in columns:

            # Column property must atleast contain name and type.
            if 'name' not in f or 'dtype' not in f:
                raise ValueError(
                    'Creating table requires column name and type at minimum.')

            field = list()

            # Get name property and append to list.
            f_name = f['name']
            field.append(f_name)

            # Get type property.
            f_dtype = f['dtype'].upper()

            if 'size' in f:
                f_size = f'({f["size"]})'
                f_dtype = f_dtype + f'({f["size"]})'
            field.append(f_dtype)

            if 'pk' in f:
                if f['pk']:
                    field.append('PRIMARY KEY')

            if 'not_null' in f:
                if f['not_null']:
                    f_not_null = 'NOT NULL'
                    field.append(f_not_null)

            if 'default' in f:
                if f['default'] is not None:
                    if f_dtype in ('CHAR', 'VARCHAR', 'TEXT'):
                        f_default = 'DEFAULT "{f["default"]}"'
                    else:
                        f_default = f'DEFAULT {f["default"]}'
                    field.append(f_default)

            field_str = ' '.join(field)

            column_strs.append(field_str)

        columns_str = ', '.join(column_strs)
        query = f'CREATE TABLE IF NOT EXISTS {table_name}( {columns_str} );'
        try:
            cursor = self._connection.cursor()
            cursor.execute(query)
        except sqlite3.Error as e:
            raise e

    def drop_table(self, table: str):
        query = f'DROP TABLE {table};'
        try:
            cursor = self._connection.cursor()
            cursor.execute(query)
        except sqlite3.Error as e:
            raise e

    def commit(self):

        try:
            self._connection.commit()
        except sqlite3.Error as e:
            raise e
