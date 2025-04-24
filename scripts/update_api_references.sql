-- SQL Migration Script to update API table references
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
