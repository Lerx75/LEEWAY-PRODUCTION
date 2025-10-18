// Minimal module declaration to satisfy TypeScript for better-sqlite3 in server-side code.
declare module 'better-sqlite3' {
  const Database: any;
  export default Database;
}
