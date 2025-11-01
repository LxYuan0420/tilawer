## what
Use UUIDv7 when you need unique identifiers that also sort by time. It keeps
inserts and queries fast because new values arrive in time order instead of
jumping around.

## context
PostgreSQL uses a B tree index for primary keys. With UUIDv4 the values are
fully random. Each insert lands in a random place in the tree. This causes page
splits, scattered pages on disk, and wasted space. At small scale this is fine.
At millions of rows, it often slows inserts and grows the index.

UUIDv7 has a time based prefix. New values are close to the previous ones in the
index. The index stays compact with fewer page splits and fewer random reads.
You keep almost the same collision protection because most bits are still
random.

PostgreSQL version 18 provides a built in function named `uuidv7()`. Earlier
versions do not have this function, but you can generate UUIDv7 in application
code or through an extension.

## steps or snippet
You can try this with PostgreSQL version 18.

```bash
brew install postgresql@18
brew services start postgresql@18
psql
```

```sql
CREATE TABLE orders (
  id uuid DEFAULT uuidv7() PRIMARY KEY,
  created_at timestamptz DEFAULT now(),
  note text
);

INSERT INTO orders (note)
SELECT 'row ' || g FROM generate_series(1, 5) AS g;

SELECT id, created_at
FROM orders
ORDER BY id
LIMIT 5;
```

## pitfalls
UUID versions share the same storage type in PostgreSQL, but they are not the
same in meaning. Switching from UUIDv4 to UUIDv7 changes ordering and index
behavior. If you start generating UUIDv7 today, the old UUIDv4 values remain and
do not sort by time. Plan your migration and rebuild indexes when you change
generation.

PostgreSQL earlier than version 18 does not include `uuidv7()`. Use a library,
an extension, or application code until you can upgrade.

Migration from UUIDv4 to UUIDv7 is generally not advised because it is not
straightforward, is time consuming and error prone, and is usually unnecessary
at low write rates (for example, around 50 requests per second).

## links
- RFC 9562 (defines UUIDv7): https://www.rfc-editor.org/rfc/rfc9562
- PostgreSQL 18 documentation: https://www.postgresql.org/docs/18/
- Why UUIDv7 helps indexes (Nile): https://www.thenile.dev/blog/uuidv7
- Background on B tree fragmentation with random keys: https://www.cockroachlabs.com/blog/ordered-uuid/
