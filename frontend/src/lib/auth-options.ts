export const authOptions = {
  providers: [],
  session: { strategy: 'jwt' as const, maxAge: 60 * 60 },
  callbacks: {
    async jwt({ token }: any) {
      return token;
    },
    async session({ session }: any) {
      return session;
    },
  },
};
