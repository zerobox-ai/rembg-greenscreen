param(
[string]$target,    
[int]$batches=5,
[int]$retry_times=5)

# compute the fps dynamically, so we don't waste compute on setting higher
$fps_fraction = ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate $target
$fps = [math]::ceiling( (Invoke-Expression $fps_fraction) )

# if the file is VBR you get crazy results like 16000
if( $fps -gt 30 -or $fps -lt 0 ){
    $fps=24
}

$orig_folder = Get-Location

$tempfolder = "f_{0}" -f (get-item $target).Name

Write-Output $tempfolder

$folder = (get-item $target).Directory.FullName


New-Item -Path "$folder" -Name "$tempfolder" -ItemType "directory" -Force
cp $target $folder/$tempfolder
Set-Location $folder/$tempfolder

# have we already exported the frames?
$skip_create_frames = (Get-ChildItem *.jpg | Measure-Object).Count -gt 10

# turn video into frames
if(-not $skip_create_frames){
    # now halfing the res of the alpha matte, gives a pretty big speedup
    ffmpeg -i $target -vf "fps=$fps,scale=iw*.5:ih*.5" %d.jpg
}else{
    Write-Output "skipping creation of frames..."
}

# retry failed ones
1..$retry_times | % { 

    Write-Output "moving pngs from old batches up"

    # copy down any previously completed pngs from other
    # runs which have been prematurely terminated
    Get-ChildItem -Directory | %{ 
        $pngs = Get-ChildItem ("./{0}/*.png" -f $_.Name)
        if( $pngs.Count -gt 0 ){  
            mv -Force ($pngs) ./  
        }
    }

    Write-Output "deleting old batches"

    $dirs = (Get-ChildItem -Directory)
    # delete any previous batch folders
    if( $dirs.Count -gt 0 ) {
        Remove-Item $dirs -Force -Recurse
    }

    Write-Output "deleting failed conversions"

    # delete any failed png conversions
    Get-ChildItem *.png | where Length -eq 0 | Remove-Item

    $d = 0;

    Write-Output "creating new batches and executing"

    # process all the frames into batches
    Get-ChildItem | 
        Where-Object Name -imatch "jpg" |
        ForEach-Object { 
            New-Object PSObject  -property @{
                PngNotPresent= -not ( Test-Path -Path ($_.name -replace ".jpg", ".out.png") ); 
                Path=$_.Name;
            }  } |
        Where-Object PngNotPresent | 
        ForEach-Object { 
            New-Object PSObject  -property @{
                PngNotPresent= $_.PngNotPresent;
                Path=$_.Path;
                Number=$x++; #we do this after filter on PngNotPresent intentionally
            }  } |
        Group-Object -Property {$_.Number % $batches} |
        ForEach-Object{
            $gn = $_.Number

            New-Item -Name "$gn" -ItemType "directory" -Force

            $_.Group | ForEach-Object{ Copy-Item $_.Path $gn }

            Start-Job -ScriptBlock {param([string]$folder, [string]$gn, [string]$tempfolder)

                if($PSVersionTable.Platform -eq "Unix"){
                    C:/Windows/System32/cmd.exe /K "$env:anaconda\\Scripts\\activate.bat $env:anaconda && cd $env:rembg && python -m src.rembg.cmd.cli -p $folder\$tempfolder\$gn 2>&1 && exit"
                }else{
                    Set-Location ~/git/rembg/
                    python -m src.rembg.cmd.cli -p $folder/$tempfolder/$gn
                }

                
                Copy-Item $folder/$tempfolder/$gn/*.png $folder
                Remove-Item $folder/$tempfolder/$gn/ -Force -Recurse

            } -ArgumentList "$folder", "$gn", "$tempfolder"

        }

    get-job | Wait-Job

    # num jpgs = num pngs
    $equal_pngs = (Get-ChildItem *.jpg | Measure-Object).Count -eq (Get-ChildItem *.png | Measure-Object).Count
    # none of the conversations failed
    $noempty_pngs = (Get-ChildItem *.png | Where-Object Length -eq 0 | Measure-Object ).Count -eq 0

    #check for every JPG file there is a corresponding non-empty PNG present
    if( $equal_pngs -and $noempty_pngs){

        echo "equal jpgs and pngs detected, and all pngs are non-zero, making mov file..."

        ffmpeg -i %d.out.png -vcodec png ($target -replace "\..+", ".mov")

        Write-Output "moving mov file back to main dir"
        Move-Item *.mov ../
        Remove-Item $tempfolder -Recurse -Force
        Set-Location $orig_folder
        
        # we good
        exit
    }

}

Write-Output "didnt detect conditions to create mov"
# go back
Set-Location $orig_folder