import Link from 'next/link';
import { useRouter } from 'next/router';
import '../styles/globals.css';

function Navigation() {
  const router = useRouter();
  
  const navItems = [
    { href: '/', label: 'Dashboard' },
    { href: '/return', label: 'Return' },
    { href: '/direction', label: 'Direction' },
    { href: '/volatility', label: 'Volatility' },
    { href: '/forecast', label: 'Forecast' },
    { href: '/regime', label: 'Regime' }
  ];

  return (
    <header className="header">
      <div className="container">
        <h1 style={{ marginBottom: '1rem', fontSize: '1.5rem', fontWeight: 'bold' }}>
          MLOps Finance Dashboard
        </h1>
        <nav className="nav">
          {navItems.map(item => (
            <Link
              key={item.href}
              href={item.href}
              className={`nav-link ${router.pathname === item.href ? 'active' : ''}`}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}

export default function App({ Component, pageProps }) {
  return (
    <>
      <Navigation />
      <main className="container">
        <Component {...pageProps} />
      </main>
    </>
  );
}
