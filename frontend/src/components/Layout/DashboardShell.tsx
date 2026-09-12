import React from 'react';

interface Props {
  sidebar: React.ReactNode;
  header: React.ReactNode;
  children: React.ReactNode;
}

export default function DashboardShell({ sidebar, header, children }: Props) {
  return (
    <div className="flex h-screen w-full relative p-2 sm:p-4 lg:p-6 lg:py-8 justify-center items-center">
      {/* Single giant glass pane containing everything */}
      <div className="flex w-full h-full max-w-[1700px] bg-white/30 backdrop-blur-xl rounded-[2rem] border border-white/40 shadow-2xl overflow-hidden shadow-blue-900/10">
        <div className="hidden lg:flex w-60 xl:w-64 flex-shrink-0 flex-col h-full border-r border-white/40">{sidebar}</div>

        <main className="flex-1 flex flex-col min-w-0 h-full overflow-hidden relative z-10">
          <div className="flex-shrink-0 relative z-20">{header}</div>
          <div className="flex-1 overflow-auto px-3 pb-6 pt-2 md:px-6 lg:px-8 styled-scroll scroll-smooth">
            <div className="max-w-[1400px] mx-auto w-full">{children}</div>
          </div>
        </main>
      </div>
    </div>
  );
}
