import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = {
  title: "India Hikes Trek Finder — Find the Perfect Trek for Your Group",
  description:
    "AI-powered trek recommendations from Indiahikes. Tell us about your group and preferences, and we'll find Himalayan treks that fit everyone."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
