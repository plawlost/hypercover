import { useState } from 'react'
import { HistoryResponse, HistoryEntry } from '@/types/history'
import { ActivityBadge } from './ActivityBadge'
import { DateFormatter } from './DateFormatter'
import { ConfirmationModal } from './ConfirmationModal'
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ChevronLeft, ChevronRight, Trash2 } from 'lucide-react'

interface HistoryListProps {
  historyData: HistoryResponse
  onClearHistory: () => Promise<void>
}

export function HistoryList({ historyData, onClearHistory }: HistoryListProps) {
  const [currentPage, setCurrentPage] = useState(1)
  const [filter, setFilter] = useState<string | undefined>()
  const [isConfirmModalOpen, setIsConfirmModalOpen] = useState(false)
  const [expandedEntry, setExpandedEntry] = useState<string | null>(null)

  const handlePageChange = (newPage: number) => {
    setCurrentPage(newPage)
    // TODO: Fetch new page data
  }

  const handleFilterChange = (value: string) => {
    setFilter(value === 'all' ? undefined : value)
    setCurrentPage(1)
    // TODO: Fetch filtered data
  }

  const handleClearHistory = async () => {
    await onClearHistory()
    setIsConfirmModalOpen(false)
  }

  const toggleEntryExpansion = (id: string) => {
    setExpandedEntry(expandedEntry === id ? null : id)
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <Select onValueChange={handleFilterChange}>
          <SelectTrigger className="w-[180px] bg-gray-800 text-white">
            <SelectValue placeholder="Filter by type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Activities</SelectItem>
            <SelectItem value="linkedin_search">LinkedIn Search</SelectItem>
            <SelectItem value="resume_upload">Resume Upload</SelectItem>
            <SelectItem value="job_suggestion">Job Suggestion</SelectItem>
            <SelectItem value="profile_update">Profile Update</SelectItem>
            <SelectItem value="subscription_change">Subscription Change</SelectItem>
            <SelectItem value="login">Login</SelectItem>
            <SelectItem value="logout">Logout</SelectItem>
          </SelectContent>
        </Select>
        <Button 
          variant="destructive" 
          onClick={() => setIsConfirmModalOpen(true)}
          className="bg-red-600 hover:bg-red-700"
        >
          <Trash2 className="mr-2 h-4 w-4" /> Clear History
        </Button>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Activity</TableHead>
            <TableHead>Date</TableHead>
            <TableHead>Details</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {historyData.history.map((entry: HistoryEntry) => (
            <TableRow key={entry.id} className="cursor-pointer hover:bg-gray-800" onClick={() => toggleEntryExpansion(entry.id)}>
              <TableCell>
                <ActivityBadge type={entry.activity_type} />
              </TableCell>
              <TableCell>
                <DateFormatter date={entry.created_at} />
              </TableCell>
              <TableCell>
                {expandedEntry === entry.id ? (
                  <pre className="whitespace-pre-wrap text-sm">
                    {JSON.stringify(entry.details, null, 2)}
                  </pre>
                ) : (
                  'Click to expand'
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <div className="flex justify-between items-center mt-4">
        <div className="text-sm text-gray-400">
          Showing {historyData.pagination.current_page} of {historyData.pagination.total_pages} pages
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === historyData.pagination.total_pages}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <ConfirmationModal
        isOpen={isConfirmModalOpen}
        onClose={() => setIsConfirmModalOpen(false)}
        onConfirm={handleClearHistory}
        title="Clear History"
        message="Are you sure you want to clear your entire activity history? This action cannot be undone."
      />
    </div>
  )
}

