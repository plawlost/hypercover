'use client'

import { useState, useEffect } from 'react'
import { Layout } from '@/components/layout'
import { HistoryList } from '@/components/history/HistoryList'
import { HistoryStats } from '@/components/history/HistoryStats'
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { historyApi } from '@/lib/historyApi'
import { HistoryResponse, HistoryStats as HistoryStatsType } from '@/types/history'

export default function HistoryPage() {
  const [activeTab, setActiveTab] = useState<'list' | 'stats'>('list')
  const [historyData, setHistoryData] = useState<HistoryResponse | null>(null)
  const [historyStats, setHistoryStats] = useState<HistoryStatsType | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      setError(null)
      try {
        const [historyResponse, statsResponse] = await Promise.all([
          historyApi.getHistory(),
          historyApi.getStats()
        ])
        setHistoryData(historyResponse)
        setHistoryStats(statsResponse)
      } catch (err) {
        setError('Failed to fetch history data. Please try again.')
        console.error(err)
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [])

  const handleClearHistory = async () => {
    try {
      await historyApi.clearHistory()
      const [historyResponse, statsResponse] = await Promise.all([
        historyApi.getHistory(),
        historyApi.getStats()
      ])
      setHistoryData(historyResponse)
      setHistoryStats(statsResponse)
    } catch (err) {
      setError('Failed to clear history. Please try again.')
      console.error(err)
    }
  }

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-white mb-6">Activity History</h1>
        
        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as 'list' | 'stats')}>
          <TabsList className="bg-gray-800 mb-6">
            <TabsTrigger value="list" className="text-white data-[state=active]:bg-[#f15b3b] data-[state=active]:text-black">List View</TabsTrigger>
            <TabsTrigger value="stats" className="text-white data-[state=active]:bg-[#f15b3b] data-[state=active]:text-black">Stats View</TabsTrigger>
          </TabsList>
          
          <TabsContent value="list">
            {isLoading ? (
              <div className="text-white">Loading history...</div>
            ) : error ? (
              <div className="text-red-500">{error}</div>
            ) : historyData ? (
              <HistoryList 
                historyData={historyData} 
                onClearHistory={handleClearHistory}
              />
            ) : null}
          </TabsContent>
          
          <TabsContent value="stats">
            {isLoading ? (
              <div className="text-white">Loading stats...</div>
            ) : error ? (
              <div className="text-red-500">{error}</div>
            ) : historyStats ? (
              <HistoryStats stats={historyStats} />
            ) : null}
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  )
}

