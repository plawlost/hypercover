import { ReactNode } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { Home, FileText, Settings, HelpCircle, LogOut, Clock } from 'lucide-react'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  const router = useRouter()

  const isActive = (path: string) => router.pathname === path

  return (
    <div className="min-h-screen bg-black text-white">
      <header className="border-b border-gray-800 bg-black/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-3">
              <svg
                viewBox="0 0 317.48 396.67"
                className="w-8 h-10"
                aria-hidden="true"
              >
                {/* ... (SVG content remains unchanged) ... */}
              </svg>
              <span className="font-monument text-white text-lg">HyperCover</span>
            </div>

            <nav className="flex items-center gap-6">
              <Link 
                href="/dashboard" 
                className={`flex items-center gap-2 transition-colors ${isActive('/dashboard') ? 'text-white' : 'text-white/60 hover:text-white'}`}
              >
                <Home className="w-4 h-4" />
                <span>Home</span>
              </Link>
              <Link 
                href="/dashboard/history" 
                className={`flex items-center gap-2 transition-colors ${isActive('/dashboard/history') ? 'text-white' : 'text-white/60 hover:text-white'}`}
              >
                <Clock className="w-4 h-4" />
                <span>History</span>
              </Link>
              <Link 
                href="/dashboard/settings" 
                className={`flex items-center gap-2 transition-colors ${isActive('/dashboard/settings') ? 'text-white' : 'text-white/60 hover:text-white'}`}
              >
                <Settings className="w-4 h-4" />
                <span>Settings</span>
              </Link>
              <Link 
                href="/help" 
                className={`flex items-center gap-2 transition-colors ${isActive('/help') ? 'text-white' : 'text-white/60 hover:text-white'}`}
              >
                <HelpCircle className="w-4 h-4" />
                <span>Help</span>
              </Link>
              <button 
                onClick={() => {/* TODO: Implement logout */}}
                className="flex items-center gap-2 text-white/60 hover:text-white transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span>Logout</span>
              </button>
            </nav>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-12 min-h-[calc(100vh-4rem)]">
        {children}
      </main>

      <footer className="border-t border-gray-800 bg-black/50 backdrop-blur-sm py-6">
        <div className="container mx-auto px-4 text-center text-white/60">
          <p>© 2023 HyperCover. All rights reserved.</p>
          <div className="mt-2 space-x-4">
            <Link href="/privacy" className="hover:text-white transition-colors">Privacy Policy</Link>
            <Link href="/terms" className="hover:text-white transition-colors">Terms of Service</Link>
          </div>
        </div>
      </footer>
    </div>
  )
}

