import { HistoryStats as HistoryStatsType } from '@/types/history'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from 'recharts'

interface HistoryStatsProps {
  stats: HistoryStatsType
}

export function HistoryStats({ stats }: HistoryStatsProps) {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Card>
        <CardHeader>
          <CardTitle>Total Activities</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-4xl font-bold">{stats.total_activities}</div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Activity Distribution</CardTitle>
        </CardHeader>
        <CardContent className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={stats.activity_counts}>
              <XAxis dataKey="activity_type" />
              <YAxis />
              <Bar dataKey="count" fill="#f15b3b" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {stats.recent_activity.map((activity, index) => (
              <li key={index} className="text-sm">
                {new Date(activity.created_at).toLocaleString()}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}

