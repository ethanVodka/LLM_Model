import { useState } from 'react'
import './App.css'
import { AppSidebar, type AppView } from './components/AppSidebar'
import { ChatPage } from './features/chat/ChatPage'
import { OverviewPage } from './features/overview/OverviewPage'

export function App() {
  const [activeView, setActiveView] = useState<AppView>('chat')
  const [isMenuOpen, setIsMenuOpen] = useState(true)

  return (
    <div className="workspace">
      <AppSidebar
        activeView={activeView}
        isOpen={isMenuOpen}
        onToggle={() => setIsMenuOpen((current) => !current)}
        onViewChange={setActiveView}
      />
      <main className="content-shell">
        {activeView === 'chat' ? <ChatPage /> : <OverviewPage />}
      </main>
    </div>
  )
}
