-- Optional future cloud history. Not applied to any account by this prototype.
-- Current app stores result-only history in the browser and never uploads to Storage.
create table if not exists public.scans (
 id uuid primary key default gen_random_uuid(),
 user_id uuid not null references auth.users(id) on delete cascade,
 created_at timestamptz not null default now(),
 model_version text not null,
 status text not null check (status in ('poor_quality','unrecognized','uncertain','healthy','suspected')),
 result jsonb not null
);
alter table public.scans enable row level security;
revoke all on table public.scans from anon, authenticated;
grant select, delete on table public.scans to authenticated;
create policy "Users read their own scan results" on public.scans for select to authenticated using ((select auth.uid()) = user_id);
create policy "Users delete their own scan results" on public.scans for delete to authenticated using ((select auth.uid()) = user_id);
-- Only a trusted backend may insert validated model results, after verifying the user JWT.
-- Never expose the service-role key to React. No direct browser INSERT/UPDATE policy.
create index if not exists scans_user_created on public.scans(user_id,created_at desc);
