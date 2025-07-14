-- IMPORTANT: Only execute this after validating the migration was successful

-- Drop the original tables from public schema
DROP TABLE IF EXISTS public.api_keys CASCADE;
DROP TABLE IF EXISTS public.api_key_audit_logs CASCADE;
DROP TABLE IF EXISTS public.api_keys_id_seq CASCADE;
DROP TABLE IF EXISTS public.api_key_audit_logs_id_seq CASCADE;

-- Clean up backup tables (if they exist)
DROP TABLE IF EXISTS public.api_keys_backup CASCADE;
DROP TABLE IF EXISTS public.api_key_audit_logs_backup CASCADE;
