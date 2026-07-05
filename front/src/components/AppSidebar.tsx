export type AppView = 'chat' | 'overview'

type AppSidebarProps = Readonly<{
  activeView: AppView
  isOpen: boolean
  onToggle: () => void
  onViewChange: (view: AppView) => void
}>

export function AppSidebar({
  activeView,
  isOpen,
  onToggle,
  onViewChange,
}: AppSidebarProps) {
  return (
    <aside className="app-sidebar" data-expanded={isOpen}>
      <div className="sidebar-header">
        <button
          className="icon-button menu-toggle"
          type="button"
          onClick={onToggle}
          aria-label={isOpen ? 'メニューを閉じる' : 'メニューを開く'}
          aria-expanded={isOpen}
        >
          <span aria-hidden="true">☰</span>
        </button>
        <span className="brand-label">Learning LLM</span>
      </div>

      <nav className="sidebar-nav" aria-label="メインメニュー">
        <button
          type="button"
          className="nav-item"
          aria-current={activeView === 'chat' ? 'page' : undefined}
          onClick={() => onViewChange('chat')}
        >
          <span className="nav-icon" aria-hidden="true">
            ◇
          </span>
          <span className="nav-label">チャット</span>
        </button>
        <button
          type="button"
          className="nav-item"
          aria-current={activeView === 'overview' ? 'page' : undefined}
          onClick={() => onViewChange('overview')}
        >
          <span className="nav-icon" aria-hidden="true">
            ◫
          </span>
          <span className="nav-label">プロジェクト概要</span>
        </button>
      </nav>

      <p className="sidebar-note">Local model · CPU</p>
    </aside>
  )
}
