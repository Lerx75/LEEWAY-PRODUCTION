// Minimal module declaration to satisfy TypeScript for bcryptjs in server-side code.
declare module 'bcryptjs' {
  const bcrypt: any;
  export default bcrypt;
}
