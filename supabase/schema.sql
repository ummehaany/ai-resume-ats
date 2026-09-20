-- =============================================================================
-- ATS Resume Scorer - Supabase schema (run once in: Dashboard -> SQL Editor)
--
-- Creates the "analyses" table used by the backend and locks it down with
-- Row Level Security (RLS) so every user can only ever see / add / delete
-- THEIR OWN rows.
--
-- The backend calls the database with the signed-in user's own access token plus
-- the public anon key (it does NOT need, and must never be given, the
-- service_role key). RLS is therefore enforced by Postgres itself.
--
-- This script only creates things; it never drops or deletes data, and it is safe
-- to run more than once.
-- =============================================================================

create table if not exists public.analyses (
    id               uuid             primary key default gen_random_uuid(),
    user_id          uuid             not null references auth.users (id) on delete cascade,
    filename         text             not null default 'resume',
    ats_score        double precision not null default 0 check (ats_score between 0 and 100),
    keyword_match    double precision not null default 0,
    missing_keywords jsonb            not null default '[]'::jsonb,
    analysis_result  jsonb            not null default '{}'::jsonb,
    created_at       timestamptz      not null default now()
);

-- History is always read as "this user's newest rows first".
create index if not exists analyses_user_created_idx
    on public.analyses (user_id, created_at desc);

-- ---- Row Level Security ------------------------------------------------------
alter table public.analyses enable row level security;

-- Least privilege: the anonymous role gets nothing; signed-in users may only
-- select / insert / delete (there is deliberately no UPDATE).
revoke all on public.analyses from anon;
revoke all on public.analyses from authenticated;
grant select, insert, delete on public.analyses to authenticated;

do $$
begin
    if not exists (select 1 from pg_policies
                   where schemaname = 'public' and tablename = 'analyses'
                     and policyname = 'analyses_select_own') then
        create policy analyses_select_own on public.analyses
            for select to authenticated
            using (auth.uid() = user_id);
    end if;

    if not exists (select 1 from pg_policies
                   where schemaname = 'public' and tablename = 'analyses'
                     and policyname = 'analyses_insert_own') then
        create policy analyses_insert_own on public.analyses
            for insert to authenticated
            with check (auth.uid() = user_id);
    end if;

    if not exists (select 1 from pg_policies
                   where schemaname = 'public' and tablename = 'analyses'
                     and policyname = 'analyses_delete_own') then
        create policy analyses_delete_own on public.analyses
            for delete to authenticated
            using (auth.uid() = user_id);
    end if;
end
$$;

-- ---- Optional: quick self-check (read-only) ----------------------------------
-- select tablename, rowsecurity from pg_tables where tablename = 'analyses';   -- rowsecurity should be true
-- select policyname, cmd from pg_policies where tablename = 'analyses';         -- 3 policies
