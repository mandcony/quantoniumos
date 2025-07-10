#!/usr/bin/env python
"""
Quantonium OS - API Tables Security Script

This script secures the API key tables by:
1. Moving them from the public schema to a secure_api schema
2. Setting proper permissions
3. Adding row-level security policies
"""

import logging
import os
import sys

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("secure-api-tables")


def get_connection():
    """Get a connection to the PostgreSQL database"""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable is not set")
        sys.exit(1)

    try:
        conn = psycopg2.connect(db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        sys.exit(1)


def secure_api_tables():
    """
    Secure the API key tables by moving them to a secure schema
    and setting appropriate permissions
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # 1. Check if tables exist in public schema
        logger.info("Checking for API tables in public schema...")
        cursor.execute(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('api_keys', 'api_key_audit_logs')
        """
        )
        tables = cursor.fetchall()

        if not tables:
            logger.info("No API tables found in public schema. Nothing to secure.")
            return True

        logger.info(f"Found {len(tables)} API tables in public schema")

        # 2. Create secure schema if it doesn't exist
        logger.info("Creating secure_api schema...")
        cursor.execute(
            """
            CREATE SCHEMA IF NOT EXISTS secure_api;
            REVOKE ALL ON SCHEMA secure_api FROM PUBLIC;
            GRANT USAGE ON SCHEMA secure_api TO neondb_owner;
        """
        )

        # 3. Backup existing data
        logger.info("Backing up existing API key data...")
        for table in tables:
            table_name = table[0]
            # Check if table has data
            cursor.execute(
                sql.SQL("SELECT COUNT(*) FROM public.{}").format(
                    sql.Identifier(table_name)
                )
            )
            count = cursor.fetchone()[0]

            if count > 0:
                logger.info(f"Table {table_name} has {count} rows")
                # Create backup table
                backup_table = f"{table_name}_backup"
                logger.info(f"Creating backup table {backup_table}...")
                cursor.execute(
                    sql.SQL(
                        """
                    CREATE TABLE IF NOT EXISTS public.{} AS 
                    SELECT * FROM public.{}
                """
                    ).format(sql.Identifier(backup_table), sql.Identifier(table_name))
                )

        # 4. Move tables to secure schema
        logger.info("Moving tables to secure_api schema...")
        for table in tables:
            table_name = table[0]
            logger.info(f"Moving {table_name} to secure_api schema...")

            # Create new tables in secure schema with same structure
            cursor.execute(
                sql.SQL(
                    """
                CREATE TABLE secure_api.{} AS SELECT * FROM public.{}
            """
                ).format(sql.Identifier(table_name), sql.Identifier(table_name))
            )

            # Get indexes and constraints
            cursor.execute(
                sql.SQL(
                    """
                SELECT
                    i.relname AS index_name,
                    pg_get_indexdef(i.oid) AS index_def
                FROM
                    pg_index x
                    JOIN pg_class i ON i.oid = x.indexrelid
                    JOIN pg_class t ON t.oid = x.indrelid
                    JOIN pg_namespace n ON n.oid = t.relnamespace
                WHERE
                    n.nspname = 'public'
                    AND t.relname = %s
            """
                ),
                (table_name,),
            )

            indexes = cursor.fetchall()

            # Recreate indexes in new schema
            for idx_name, idx_def in indexes:
                # Modify the index definition to point to the new schema
                new_idx_def = idx_def.replace(
                    f"public.{table_name}", f"secure_api.{table_name}"
                )
                logger.info(f"Creating index: {new_idx_def}")
                cursor.execute(new_idx_def)

            # Add sequence if needed
            if table_name.endswith("_seq"):
                logger.info(f"Setting sequence ownership for {table_name}...")
                related_table = table_name.replace("_id_seq", "")
                cursor.execute(
                    sql.SQL(
                        """
                    ALTER SEQUENCE secure_api.{} OWNED BY secure_api.{}.id
                """
                    ).format(sql.Identifier(table_name), sql.Identifier(related_table))
                )

        # 5. Add security policies to new tables
        logger.info("Adding row-level security policies...")
        cursor.execute(
            """
            ALTER TABLE secure_api.api_keys ENABLE ROW LEVEL SECURITY;
            ALTER TABLE secure_api.api_key_audit_logs ENABLE ROW LEVEL SECURITY;
            
            -- Default policies: no access
            CREATE POLICY api_keys_policy ON secure_api.api_keys
              USING (FALSE);
              
            CREATE POLICY api_audit_policy ON secure_api.api_key_audit_logs
              USING (FALSE);
              
            -- Grant selective permissions to trusted roles only
            GRANT SELECT, INSERT, UPDATE, DELETE ON secure_api.api_keys TO neondb_owner;
            GRANT SELECT, INSERT, UPDATE, DELETE ON secure_api.api_key_audit_logs TO neondb_owner;
        """
        )

        # 6. Update application code references
        logger.info("Generating SQL migration script for application...")
        with open("scripts/update_api_references.sql", "w") as f:
            f.write(
                """-- SQL Migration Script to update API table references
-- Execute this if you need to revert the schema change
-- Replace public with secure_api in your application code

-- Tables were moved from public to secure_api schema

-- To reference the new tables:
-- OLD: public.api_keys
-- NEW: secure_api.api_keys

-- OLD: public.api_key_audit_logs
-- NEW: secure_api.api_key_audit_logs

-- OLD: public.api_keys_id_seq
-- NEW: secure_api.api_keys_id_seq

-- OLD: public.api_key_audit_logs_id_seq
-- NEW: secure_api.api_key_audit_logs_id_seq

-- Create views if necessary for backward compatibility
CREATE OR REPLACE VIEW public.api_keys AS 
    SELECT * FROM secure_api.api_keys
    WITH CHECK OPTION;

CREATE OR REPLACE VIEW public.api_key_audit_logs AS 
    SELECT * FROM secure_api.api_key_audit_logs
    WITH CHECK OPTION;

-- GRANT permissions on views
GRANT SELECT ON public.api_keys TO neondb_owner;
GRANT SELECT ON public.api_key_audit_logs TO neondb_owner;
"""
            )

        # 7. Create drop script for original tables (to be executed manually after validation)
        with open("scripts/drop_original_api_tables.sql", "w") as f:
            f.write(
                """-- IMPORTANT: Only execute this after validating the migration was successful

-- Drop the original tables from public schema
DROP TABLE IF EXISTS public.api_keys CASCADE;
DROP TABLE IF EXISTS public.api_key_audit_logs CASCADE;
DROP TABLE IF EXISTS public.api_keys_id_seq CASCADE;
DROP TABLE IF EXISTS public.api_key_audit_logs_id_seq CASCADE;

-- Clean up backup tables (if they exist)
DROP TABLE IF EXISTS public.api_keys_backup CASCADE;
DROP TABLE IF EXISTS public.api_key_audit_logs_backup CASCADE;
"""
            )

        logger.info("API tables have been secured successfully!")
        logger.info(
            "Generated SQL migration scripts in scripts/update_api_references.sql"
        )
        logger.info("Generated cleanup script in scripts/drop_original_api_tables.sql")

        # 8. Print instructions
        print("\nSECURITY UPDATE COMPLETED")
        print("-----------------------")
        print(
            "API key tables have been secured by moving them to a private schema with row-level security."
        )
        print("\nNext steps:")
        print(
            "1. Update your application code to reference secure_api.api_keys instead of public.api_keys"
        )
        print("2. Validate that the application works correctly with the new schema")
        print(
            "3. Run scripts/drop_original_api_tables.sql to remove the original tables once verified"
        )
        print("\nIf you encounter any issues, you can:")
        print("- Use the backup tables created in the public schema")
        print(
            "- Run scripts/update_api_references.sql to create views for backward compatibility"
        )

        return True

    except Exception as e:
        logger.error(f"Error securing API tables: {str(e)}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    print("=" * 80)
    print(" Quantonium OS - API Tables Security Script")
    print("=" * 80)

    if secure_api_tables():
        print("\nAPI tables have been secured successfully!")
        sys.exit(0)
    else:
        print("\nFailed to secure API tables. Check the logs for details.")
        sys.exit(1)
