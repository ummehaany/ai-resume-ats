-- =============================================================================
-- ATS Resume Scorer - Row Level Security self-test
-- Run in: Supabase Dashboard -> SQL Editor, AFTER supabase/schema.sql.
--
-- What it does: inside ONE transaction it creates two throw-away users
-- (rls-selftest-a/b@invalid.example), then acts as each of them (and as the
-- anonymous role) and checks that
--   * a user can insert / read / delete their OWN rows,
--   * a user can NOT read, delete or forge rows that belong to someone else,
--   * nobody can UPDATE a row, and the anon role can not touch the table.
-- Everything is ROLLED BACK at the end: no test users or rows are kept and no
-- existing data is read or changed (other users' rows are invisible to the test
-- users by design).
--
-- Result: the last line shows  'ALL RLS CHECKS PASSED'.  If any check fails the
-- script stops with an error that says which one - do NOT deploy in that case.
-- =============================================================================

begin;

do $$
declare
    uid_a uuid := gen_random_uuid();
    uid_b uuid := gen_random_uuid();
    aid   uuid;
    n     integer;
begin
    insert into auth.users (id, aud, role, email)
    values (uid_a, 'authenticated', 'authenticated', 'rls-selftest-a@invalid.example'),
           (uid_b, 'authenticated', 'authenticated', 'rls-selftest-b@invalid.example');

    -- ---- act as user A ------------------------------------------------------
    perform set_config('request.jwt.claims',
                       json_build_object('sub', uid_a, 'role', 'authenticated')::text, true);
    set local role authenticated;

    insert into public.analyses (user_id, filename, ats_score)
    values (uid_a, 'a.pdf', 50) returning id into aid;

    select count(*) into n from public.analyses;
    if n <> 1 then raise exception 'FAIL: user A should see exactly their 1 row, saw %', n; end if;

    begin
        insert into public.analyses (user_id, filename) values (uid_b, 'forged.pdf');
        raise exception 'FAIL: user A was able to insert a row owned by user B';
    exception when insufficient_privilege then null;   -- expected: RLS WITH CHECK rejects it
    end;

    begin
        update public.analyses set ats_score = 100 where id = aid;
        raise exception 'FAIL: UPDATE is allowed (it should not be granted)';
    exception when insufficient_privilege then null;   -- expected: no UPDATE privilege
    end;

    -- ---- act as user B ------------------------------------------------------
    reset role;
    perform set_config('request.jwt.claims',
                       json_build_object('sub', uid_b, 'role', 'authenticated')::text, true);
    set local role authenticated;

    select count(*) into n from public.analyses;
    if n <> 0 then raise exception 'FAIL: user B can see % row(s) of other users', n; end if;

    delete from public.analyses where id = aid;
    get diagnostics n = row_count;
    if n <> 0 then raise exception 'FAIL: user B deleted user A''s row'; end if;

    -- ---- the anonymous (not signed in) role ---------------------------------
    reset role;
    set local role anon;
    begin
        perform count(*) from public.analyses;
        raise exception 'FAIL: the anon role can read the table';
    exception when insufficient_privilege then null;   -- expected
    end;

    -- ---- back to user A: the row is still there and A can delete it ---------
    reset role;
    perform set_config('request.jwt.claims',
                       json_build_object('sub', uid_a, 'role', 'authenticated')::text, true);
    set local role authenticated;

    select count(*) into n from public.analyses where id = aid;
    if n <> 1 then raise exception 'FAIL: user A lost their row'; end if;

    delete from public.analyses where id = aid;
    get diagnostics n = row_count;
    if n <> 1 then raise exception 'FAIL: user A could not delete their own row'; end if;

    reset role;
end
$$;

rollback;   -- nothing above is kept

select 'ALL RLS CHECKS PASSED' as result;
